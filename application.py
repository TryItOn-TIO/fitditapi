import uvicorn
from fastapi import FastAPI, HTTPException
from datetime import datetime
import os

from .schemas import AvatarCreateRequest, AvatarCreateResponse, AvatarTryOnRequest, AvatarTryOnResponse
from .config import settings
from . import s3_handler
from . import vton_service

# --- FastAPI 애플리케이션 생성 ---
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
        # 실제 프로덕션에서는 서버를 종료하거나 에러 상태로 전환해야 합니다.
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
        model_image = s3_handler.download_image_from_s3(str(request.tryOnImgUrl))

        # 2. 상의 마스크 및 포즈 생성
        # 포즈는 한 번만 생성하면 되므로 상의 마스크 생성 시 함께 얻습니다.
        upper_mask, pose_image = vton_service.create_mask_and_pose(model_image, "Upper-body")

        # 3. 하의 마스크 생성 (동일한 모델 이미지 사용)
        lower_mask, _ = vton_service.create_mask_and_pose(model_image, "Lower-body")

        # 4. 생성된 이미지들을 S3에 업로드
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # S3 키 생성
        base_key = f"users/{request.userId}/{timestamp}"
        pose_key = f"{base_key}_pose.png"
        upper_mask_key = f"{base_key}_upper_mask.png"
        lower_mask_key = f"{base_key}_lower_mask.png"
        
        # S3 업로드 실행
        pose_url = s3_handler.upload_pil_image_to_s3(pose_image, pose_key)
        upper_mask_url = s3_handler.upload_pil_image_to_s3(upper_mask, upper_mask_key)
        lower_mask_url = s3_handler.upload_pil_image_to_s3(lower_mask, lower_mask_key)

        print("✅ 마스크 및 포즈 생성/업로드 완료.")
        return AvatarCreateResponse(
            tryOnImgUrl=request.tryOnImgUrl,
            poseImgUrl=pose_url,
            upperMaskImgUrl=upper_mask_url,
            lowerMaskImgUrl=lower_mask_url
        )

    except Exception as e:
        print(f"❌ /generate 오류: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during mask generation: {e}")


@app.post(
    "/tryon",
    response_model=AvatarTryOnResponse,
    summary="Try-On 이미지 생성",
    description="베이스 이미지, 의상, 마스크, 포즈 이미지 URL을 받아 Try-On 이미지를 생성하고 S3 URL을 반환합니다."
)
async def try_on_avatar(request: AvatarTryOnRequest):
    print(f"✅ /tryon 요청 수신: userId={request.userId}")
    try:
        # 1. S3에서 필요한 모든 이미지 다운로드
        print("⏳ 필요한 이미지들을 S3에서 다운로드 중...")
        base_image = s3_handler.download_image_from_s3(str(request.baseImgUrl))
        garment_image = s3_handler.download_image_from_s3(str(request.garmentImgUrl))
        mask_image = s3_handler.download_image_from_s3(str(request.maskImgUrl))
        pose_image = s3_handler.download_image_from_s3(str(request.poseImgUrl))
        print("✅ 이미지 다운로드 완료.")

        # 2. 가상 피팅 수행
        result_image = vton_service.perform_try_on(
            base_image=base_image,
            garment_image=garment_image,
            mask_image=mask_image,
            pose_image=pose_image
        )

        # 3. 결과 이미지를 S3에 업로드
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_key = f"results/{request.userId}/tryon_{timestamp}.png"
        result_url = s3_handler.upload_pil_image_to_s3(result_image, result_key)
        
        print("✅ Try-On 이미지 생성/업로드 완료.")
        return AvatarTryOnResponse(tryOnImgUrl=result_url)

    except Exception as e:
        print(f"❌ /tryon 오류: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during try-on: {e}")

# --- 서버 실행 (로컬 테스트용) ---
if __name__ == "__main__":
    # Docker 환경에서는 이 부분이 직접 실행되지 않고, CMD 명령어가 uvicorn을 실행합니다.
    uvicorn.run(app, host="0.0.0.0", port=8000)
