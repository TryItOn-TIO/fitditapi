"""
FitDiT 모델을 사용하여 가상 피팅(Virtual Try-on)을 수행하는 메인 스크립트.
FastAPI 서비스에 맞게 수정된 버전.
"""
import os
import torch
import numpy as np
from PIL import Image
import math
import random
from typing import Tuple

# 소스 코드가 마운트된 /app 폴더를 기준으로 모듈 임포트
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection
from src.pose_guider import PoseGuider
from src.utils_mask import get_mask_location
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
from config import settings

# --- 모델 클래스 및 헬퍼 함수 (기존 코드와 거의 동일) ---

class FitDiTGenerator:
    """FitDiT 모델과 관련 프로세서를 초기화하고 관리하는 클래스"""
    def __init__(self, config):
        self.device = config.DEVICE
        self.config = config
        weight_dtype = torch.float16 if config.WITH_FP16 else torch.bfloat16

        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(config.MAIN_MODEL_PATH, "transformer_garm"), torch_dtype=weight_dtype)
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(config.MAIN_MODEL_PATH, "transformer_vton"), torch_dtype=weight_dtype)
        
        pose_guider = PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
        pose_guider_weights = torch.load(os.path.join(config.MAIN_MODEL_PATH, "pose_guider", "diffusion_pytorch_model.bin"), weights_only=True)
        pose_guider.load_state_dict(pose_guider_weights)
        
        pose_guider.to(self.device, dtype=weight_dtype)

        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(os.path.join(config.MODEL_ROOT, "clip-vit-large-patch14"), torch_dtype=weight_dtype)
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(os.path.join(config.MODEL_ROOT, "CLIP-ViT-bigG-14-laion2B-39B-b160k"), torch_dtype=weight_dtype)

        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            config.MAIN_MODEL_PATH,
            torch_dtype=weight_dtype,
            transformer_garm=transformer_garm,
            transformer_vton=transformer_vton,
            pose_guider=pose_guider,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG
        )

        device_for_preprocessors = 'cpu' if config.OFFLOAD or config.AGGRESSIVE_OFFLOAD else self.device
        self.dwprocessor = DWposeDetector(model_root=config.MAIN_MODEL_PATH, device=device_for_preprocessors)
        self.parsing_model = Parsing(model_root=config.MAIN_MODEL_PATH, device=device_for_preprocessors)

        if config.OFFLOAD:
            self.pipeline.enable_model_cpu_offload()
        elif config.AGGRESSIVE_OFFLOAD:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.to(self.device)

# --- 헬퍼 함수들 ---
def _pad_and_resize(im, new_width, new_height, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    # ... (기존 코드와 동일)
    old_width, old_height = im.size
    ratio = min(new_width / old_width, new_height / old_height)
    new_size = (round(old_width * ratio), round(old_height * ratio))
    im_resized = im.resize(new_size, mode)
    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)
    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    new_im.paste(im_resized, (pad_w, pad_h))
    return new_im, pad_w, pad_h

def _unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    # ... (기존 코드와 동일)
    width, height = padded_im.size
    left, top = pad_w, pad_h
    right, bottom = width - pad_w, height - pad_h
    cropped_im = padded_im.crop((left, top, right, bottom))
    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)
    return resized_im

def _resize_image(img, target_size=768):
    # ... (기존 코드와 동일)
    width, height = img.size
    scale = target_size / min(width, height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    return img.resize((new_width, new_height), Image.LANCZOS)

# --- 핵심 기능 함수 (FastAPI 서비스에 맞게 수정) ---

# 모델 인스턴스를 전역으로 관리 (main.py에서 초기화)
generator: FitDiTGenerator = None

def create_mask_and_pose(vton_img: Image.Image, category: str) -> Tuple[Image.Image, Image.Image]:
    """모델 이미지에 대한 마스크와 포즈를 생성하고 PIL Image 객체로 반환합니다."""
    global generator
    if generator is None:
        raise RuntimeError("Model is not initialized.")

    print(f"Creating mask and pose for category: {category}...")
    
    with torch.inference_mode():
        vton_img_det = _resize_image(vton_img)
        pose_image_np, keypoints, _, candidate = generator.dwprocessor(np.array(vton_img_det)[:, :, ::-1])
        candidate = candidate[0]
        candidate[:, 0] *= vton_img_det.width
        candidate[:, 1] *= vton_img_det.height

        pose_image = Image.fromarray(pose_image_np[:, :, ::-1])
        model_parse, _ = generator.parsing_model(vton_img_det)

        mask, _ = get_mask_location(
            category, model_parse, candidate,
            model_parse.width, model_parse.height,
            0, 0, 0, 0
        )
        mask = mask.resize(vton_img.size, Image.NEAREST).convert("L")

    print("Mask and pose creation complete.")
    return mask, pose_image

def perform_try_on(
    base_image: Image.Image, 
    garment_image: Image.Image, 
    mask_image: Image.Image, 
    pose_image: Image.Image
) -> Image.Image:
    """주어진 이미지들을 사용하여 가상 피팅을 수행하고 결과 이미지를 반환합니다."""
    global generator
    if generator is None:
        raise RuntimeError("Model is not initialized.")

    print("Performing virtual try-on...")
    
    new_width, new_height = map(int, settings.TRYON_RESOLUTION.split("x"))
    
    with torch.inference_mode():
        model_image_size = base_image.size
        garm_img_resized, _, _ = _pad_and_resize(garment_image, new_width, new_height)
        vton_img_resized, pad_w, pad_h = _pad_and_resize(base_image, new_width, new_height)
        mask_resized, _, _ = _pad_and_resize(mask_image, new_width, new_height, pad_color=(0,0,0))
        mask_resized = mask_resized.convert("L")
        pose_image_resized, _, _ = _pad_and_resize(pose_image, new_width, new_height, pad_color=(0,0,0))

        seed = settings.SEED
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        print(f"Using seed: {seed}")

        result_images = generator.pipeline(
            height=new_height,
            width=new_width,
            guidance_scale=settings.GUIDANCE_SCALE,
            num_inference_steps=settings.STEPS,
            generator=torch.Generator(settings.DEVICE).manual_seed(seed),
            cloth_image=garm_img_resized,
            model_image=vton_img_resized,
            mask=mask_resized,
            pose_image=pose_image_resized,
            num_images_per_prompt=1
        ).images

        final_image = _unpad_and_resize(result_images[0], pad_w, pad_h, model_image_size[0], model_image_size[1])
    
    print("Virtual try-on complete.")
    return final_image
