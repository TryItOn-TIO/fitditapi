"""
FitDiT 모델을 사용하여 가상 피팅(Virtual Try-on)을 수행하는 메인 스크립트.
개발 환경에 최적화된 버전으로, Docker 컨테이너에서 실행되며
소스 코드와 모델은 외부 호스트 머신에서 볼륨으로 마운트됩니다.
"""
import os
import torch
import numpy as np
from PIL import Image
import math
import random

# 소스 코드가 마운트된 /app 폴더를 기준으로 모듈 임포트
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection
from src.pose_guider import PoseGuider
from src.utils_mask import get_mask_location
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton

# --- 설정 부 ---

def get_config():
    """리소스 및 실행 관련 설정을 반환합니다."""
    return {
        # docker run에서 마운트될 경로. 컨테이너 내부의 절대 경로를 사용합니다.
        "model_root": "/models",
        "main_model_path": "/models/FitDiT",
        "device": "cuda:0",
        "with_fp16": True,
        "offload": False,
        "aggressive_offload": True,
        "tryon_resolution": "1152x1536",
        "steps": 20,
        "guidance_scale": 2.0,
        "seed": -1,
    }

class FitDiTGenerator:
    """FitDiT 모델과 관련 프로세서를 초기화하고 관리하는 클래스"""
    def __init__(self, config):
        self.device = config["device"]
        self.config = config
        weight_dtype = torch.float16 if config["with_fp16"] else torch.bfloat16

        # 모델 컴포넌트 로드
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(config["main_model_path"], "transformer_garm"), torch_dtype=weight_dtype)
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(config["main_model_path"], "transformer_vton"), torch_dtype=weight_dtype)
        
        pose_guider = PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
        pose_guider_weights = torch.load(os.path.join(config["main_model_path"], "pose_guider", "diffusion_pytorch_model.bin"), weights_only=True)
        pose_guider.load_state_dict(pose_guider_weights)
        
        # pose_guider를 다른 모델들과 동일한 데이터 타입 및 장치로 이동
        pose_guider.to(self.device, dtype=weight_dtype)

        # image_encoder들은 각각의 폴더에서 로드
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(os.path.join(config["model_root"], "clip-vit-large-patch14"), torch_dtype=weight_dtype)
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(os.path.join(config["model_root"], "CLIP-ViT-bigG-14-laion2B-39B-b160k"), torch_dtype=weight_dtype)

        # 전체 파이프라인 초기화
        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            config["main_model_path"],
            torch_dtype=weight_dtype,
            transformer_garm=transformer_garm,
            transformer_vton=transformer_vton,
            pose_guider=pose_guider,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG
        )

        device_for_preprocessors = 'cpu' if config["offload"] or config["aggressive_offload"] else self.device
        # 전처리기 모델들은 main_model_path에서 찾도록 설정
        self.dwprocessor = DWposeDetector(model_root=config["main_model_path"], device=device_for_preprocessors)
        self.parsing_model = Parsing(model_root=config["main_model_path"], device=device_for_preprocessors)

        # CPU 오프로드 설정
        if config["offload"]:
            self.pipeline.enable_model_cpu_offload()
        elif config["aggressive_offload"]:
            self.pipeline.enable_sequential_cpu_offload()
        else:
            self.pipeline.to(self.device)


# --- 헬퍼 함수 ---

def _pad_and_resize(im, new_width, new_height, pad_color=(255, 255, 255), mode=Image.LANCZOS):
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
    width, height = padded_im.size
    left, top = pad_w, pad_h
    right, bottom = width - pad_w, height - pad_h
    cropped_im = padded_im.crop((left, top, right, bottom))
    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)
    return resized_im

def _resize_image(img, target_size=768):
    width, height = img.size
    scale = target_size / min(width, height)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    return img.resize((new_width, new_height), Image.LANCZOS)


# --- 핵심 기능 함수 ---

def load_image(image_path: str) -> Image.Image:
    """로컬 경로에서 이미지를 불러옵니다."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    print(f"이미지 로드: {image_path}")
    return Image.open(image_path).convert("RGB")

def generate_and_save_mask(generator: FitDiTGenerator, vton_img: Image.Image, category: str, output_dir: str = "output"):
    """모델 이미지에 대한 마스크와 포즈를 생성하고 로컬에 저장합니다."""
    print("마스크 및 포즈 생성 시작...")
    os.makedirs(output_dir, exist_ok=True)

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
            0, 0, 0, 0  # offset_top, offset_bottom, offset_left, offset_right
        )
        mask = mask.resize(vton_img.size, Image.NEAREST).convert("L")

    mask_path = os.path.join(output_dir, "mask.png")
    pose_path = os.path.join(output_dir, "pose.png")
    mask.save(mask_path)
    pose_image.save(pose_path)
    print(f"마스크 저장 완료: {mask_path}")
    print(f"포즈 이미지 저장 완료: {pose_path}")

    return mask_path, pose_path

def perform_try_on(generator: FitDiTGenerator, model_image: Image.Image, garment_image: Image.Image, mask_path: str, pose_path: str, config: dict):
    """주어진 이미지와 마스크, 포즈 정보를 사용하여 가상 피팅을 수행합니다."""
    print("가상 피팅 시작...")
    mask = Image.open(mask_path).convert("L")
    pose_image = Image.open(pose_path)

    new_width, new_height = map(int, config["tryon_resolution"].split("x"))
    
    with torch.inference_mode():
        model_image_size = model_image.size
        garm_img_resized, _, _ = _pad_and_resize(garment_image, new_width, new_height)
        vton_img_resized, pad_w, pad_h = _pad_and_resize(model_image, new_width, new_height)
        mask_resized, _, _ = _pad_and_resize(mask, new_width, new_height, pad_color=(0,0,0))
        mask_resized = mask_resized.convert("L")
        pose_image_resized, _, _ = _pad_and_resize(pose_image, new_width, new_height, pad_color=(0,0,0))

        seed = config["seed"]
        if seed == -1:
            seed = random.randint(0, 2147483647)
        
        print(f"사용된 시드: {seed}")

        result_images = generator.pipeline(
            height=new_height,
            width=new_width,
            guidance_scale=config["guidance_scale"],
            num_inference_steps=config["steps"],
            generator=torch.Generator(config['device']).manual_seed(seed),
            cloth_image=garm_img_resized,
            model_image=vton_img_resized,
            mask=mask_resized,
            pose_image=pose_image_resized,
            num_images_per_prompt=1
        ).images

        final_image = _unpad_and_resize(result_images[0], pad_w, pad_h, model_image_size[0], model_image_size[1])
    
    print("가상 피팅 완료.")
    return final_image


# --- 메인 실행 로직 ---

if __name__ == "__main__":
    try:
        # 1. 설정 불러오기
        config = get_config()
        
        # --- 사용자가 지정해야 할 부분 ---
        # 이 경로들은 컨테이너 내부의 /app 폴더를 기준으로 합니다.
        # docker run 명령어에서 -v 옵션으로 호스트의 폴더와 연결됩니다.
        MODEL_IMAGE_PATH = "images/model_person.jpg"
        GARMENT_IMAGE_PATH = "images/garment_top.png"
        OUTPUT_DIR = "results"
        # 카테고리 지정: "Upper-body", "Lower-body", "Dresses" 중 하나 선택
        GARMENT_CATEGORY = "Upper-body"
        # ---------------------------------

        # 2. 모델 초기화
        print("FitDiT 모델을 초기화하는 중...")
        generator = FitDiTGenerator(config)
        print("모델 초기화 완료.")

        # 3. 모델 이미지 불러오기
        model_image = load_image(MODEL_IMAGE_PATH)

        # 4. 마스크 생성 및 저장
        mask_path, pose_path = generate_and_save_mask(generator, model_image, GARMENT_CATEGORY, OUTPUT_DIR)
        
        # 5. 의상 이미지 불러오기
        garment_image = load_image(GARMENT_IMAGE_PATH)
        
        # 6. 가상 피팅 실행
        try_on_result = perform_try_on(generator, model_image, garment_image, mask_path, pose_path, config)
        
        # 7. 결과 저장
        # 입력 파일명을 기반으로 결과 파일명 생성
        model_basename = os.path.splitext(os.path.basename(MODEL_IMAGE_PATH))[0]
        garment_basename = os.path.splitext(os.path.basename(GARMENT_IMAGE_PATH))[0]
        result_filename = f"try_on_result_{model_basename}_with_{garment_basename}.png"
        result_path = os.path.join(OUTPUT_DIR, result_filename)
        try_on_result.save(result_path)
        print(f"최종 결과 저장 완료: {result_path}")
        
    except FileNotFoundError as e:
        print(f"오류: {e}. 'images' 폴더에 이미지가 있는지, 또는 파일 경로가 올바른지 확인하세요.")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")