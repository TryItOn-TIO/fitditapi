# application.py

import uvicorn
from fastapi import FastAPI, HTTPException
from datetime import datetime
import os
import asyncio
import traceback

from schemas import AvatarCreateRequest, AvatarCreateResponse, AvatarTryOnRequest, AvatarTryOnResponse
from config import settings
import s3_handler
import vton_service

# ... (FastAPI 앱 생성, startup 이벤트 핸들러는 그대로 유지) ...
app = FastAPI(
    title="가상 피팅(Virtual Try-On) API",
    description="FitDiT 모델을 사용하여 상/하의 가상 피팅을 수행합니다.",
    version="1.0.0"
)

# --- 모델 생명주기 관리 ---
@app.on_event("startup")
def startup_event():
    """서버 시작 시 ML 모델을 로드합니다."""
    print("🚀 FastAPI server starting up...")
    try:
        print("⏳ Loading FitDiT model, this may take a while...")
        vton_service.generator = vton_service.FitDiTGenerator(settings)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError("ML model could not be loaded!")

# --- API 엔드포인트 ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the FitDiT Virtual Try-On API"}

@app.post(
    "/generate",
    response_model=AvatarCreateResponse,
    summary="Pose 및 상/하의 Mask 이미지 생성",
    description="원본 이미지 S3 주소를 받아, Pose와 상/하의 Mask 이미지를 생성하고 각각의 S3 주소를 반환합니다."
)
async def generate_images(request: AvatarCreateRequest):
    print(f"✅ /generate 요청 수신: userId={request.userId}, url={request.tryOnImgUrl}")
    try:
        # 1. S3에서 원본 이미지 다운로드
        model_image = await s3_handler.download_image_from_s3(str(request.tryOnImgUrl))

        # 2. 포즈 생성 (무거운 작업) -> 1번만 실행
        print("⏳ Pose 데이터를 생성합니다... (1번만 실행)")
        pose_image, candidate, vton_img_det = await asyncio.to_thread(vton_service.create_pose_data, model_image)

        # 3. 상의/하의 마스크를 병렬로 생성
        print("⏳ 상의/하의 Mask를 병렬로 생성합니다...")
        mask_tasks = [
            asyncio.to_thread(vton_service.create_mask_only, model_image, vton_img_det, candidate, "Upper-body"),
            asyncio.to_thread(vton_service.create_mask_only, model_image, vton_img_det, candidate, "Lower-body")
        ]
        upper_mask, lower_mask = await asyncio.gather(*mask_tasks)
        print("✅ 모든 마스크 병렬 생성 완료.")

        # 4. 생성된 모든 이미지를 S3에 병렬로 업로드
        # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_key = f"users/{request.userId}/"
        
        upload_tasks = [
            s3_handler.upload_pil_image_to_s3(pose_image, f"{base_key}pose.png"),
            s3_handler.upload_pil_image_to_s3(upper_mask, f"{base_key}upper_mask.png"),
            s3_handler.upload_pil_image_to_s3(lower_mask, f"{base_key}lower_mask.png")
        ]
        pose_url, upper_mask_url, lower_mask_url = await asyncio.gather(*upload_tasks)

        print("✅ 마스크 및 포즈 생성/업로드 완료.")
        return AvatarCreateResponse(
            tryOnImgUrl=request.tryOnImgUrl,
            poseImgUrl=pose_url,
            upperMaskImgUrl=upper_mask_url,
            lowerMaskImgUrl=lower_mask_url
        )

    except Exception as e:
        print(f"❌ /generate 오류: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during mask generation: {e}")

# ... (/tryon 엔드포인트는 이전과 동일하게 유지) ...
@app.post(
    "/tryon",
    response_model=AvatarTryOnResponse,
    summary="Try-On 이미지 생성",
    description="베이스 이미지, 의상, 마스크, 포즈 이미지 URL을 받아 Try-On 이미지를 생성하고 S3 URL을 반환합니다."
)
async def try_on_avatar(request: AvatarTryOnRequest):
    print(f"✅ /tryon 요청 수신: userId={request.userId}")
    try:
        # 1. S3에서 필요한 모든 이미지를 병렬로 다운로드 (asyncio.gather 사용)
        print("⏳ 필요한 이미지들을 병렬로 다운로드 중...")
        download_tasks = [
            s3_handler.download_image_from_s3(str(request.baseImgUrl)),
            s3_handler.download_image_from_s3(str(request.garmentImgUrl)),
            s3_handler.download_image_from_s3(str(request.maskImgUrl)),
            s3_handler.download_image_from_s3(str(request.poseImgUrl))
        ]
        base_image, garment_image, mask_image, pose_image = await asyncio.gather(*download_tasks)
        print("✅ 모든 이미지 다운로드 완료.")

        # 2. 무거운 AI 작업을 별도 스레드에서 실행 (asyncio.to_thread 사용)
        print("⏳ Try-On 모델 실행 중... (별도 스레드에서 처리)")
        result_image = await asyncio.to_thread(
            vton_service.perform_try_on,
            base_image=base_image,
            garment_image=garment_image,
            mask_image=mask_image,
            pose_image=pose_image
        )

        # 3. 결과 이미지를 S3에 비동기 업로드 (await 사용)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_key = f"results/{request.userId}/tryon_{timestamp}.png"
        result_url = await s3_handler.upload_pil_image_to_s3(result_image, result_key)
        
        print("✅ Try-On 이미지 생성/업로드 완료.")
        return AvatarTryOnResponse(tryOnImgUrl=result_url)

    except Exception as e:
        print(f"❌ /tryon 오류: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred during try-on: {e}")

if __name__ == "__main__":
    uvicorn.run("application:app", host="0.0.0.0", port=8000, reload=True)