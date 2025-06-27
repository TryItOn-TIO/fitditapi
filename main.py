import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import time
import random

# --- Pydantic 모델 정의 ---

class AvatarCreateRequest(BaseModel):
    tryOnImgUrl: HttpUrl
    userId: int

class AvatarCreateResponse(BaseModel):
    tryOnImgUrl: HttpUrl
    poseImgUrl: HttpUrl
    lowerMaskImgUrl: HttpUrl
    upperMaskImgUrl: HttpUrl

class AvatarTryOnRequest(BaseModel):
    baseImgUrl: HttpUrl
    maskImgUrl: HttpUrl
    poseImgUrl: HttpUrl
    userId: int

class AvatarTryOnResponse(BaseModel):
    tryOnImgUrl: HttpUrl

# --- FastAPI 애플리케이션 생성 ---

app = FastAPI(
    title="아바타 이미지 생성 API",
    description="원본 이미지를 받아 Pose와 Mask 이미지를 생성하거나 Try-On 이미지를 생성합니다.",
    version="1.0.0"
)

# --- /generate 엔드포인트 ---

@app.post(
    "/generate",
    response_model=AvatarCreateResponse,
    summary="Pose 및 Mask 이미지 생성",
    description="원본 이미지 S3 주소를 받아, Pose와 Mask 이미지를 생성하고 각각의 S3 주소를 반환합니다."
)
async def generate_images(request: AvatarCreateRequest):
    print(f"✅ 요청 수신: TryOn 이미지 URL = {request.tryOnImgUrl}, 사용자 ID = {request.userId}")
    time.sleep(1)
    time.sleep(2)
    time.sleep(1)

    file_id = random.randint(10000, 99999)
    base_s3_url = "https://your-s3-bucket.s3.ap-northeast-2.amazonaws.com/avatars"
    
    pose_image_s3_url = f"{base_s3_url}/{file_id}_pose.jpg"
    upper_mask_image_s3_url = f"{base_s3_url}/{file_id}_upper_mask.png"
    lower_mask_image_s3_url = f"{base_s3_url}/{file_id}_lower_mask.png"
    
    print(f"✅ 생성 완료: Pose 이미지 URL = {pose_image_s3_url}")
    print(f"✅ 생성 완료: Upper Mask 이미지 URL = {upper_mask_image_s3_url}")
    print(f"✅ 생성 완료: Lower Mask 이미지 URL = {lower_mask_image_s3_url}")

    return AvatarCreateResponse(
        tryOnImgUrl=request.tryOnImgUrl,
        poseImgUrl=pose_image_s3_url,
        upperMaskImgUrl=upper_mask_image_s3_url,
        lowerMaskImgUrl=lower_mask_image_s3_url
    )

# --- /tryon 엔드포인트 ---

@app.post(
    "/tryon",
    response_model=AvatarTryOnResponse,
    summary="Try-On 이미지 생성",
    description="Base 이미지, Mask 이미지, Pose 이미지를 기반으로 Try-On 이미지를 생성하고 반환합니다."
)
async def try_on_avatar(request: AvatarTryOnRequest):
    print(f"✅ Try-On 요청 수신: base={request.baseImgUrl}, mask={request.maskImgUrl}, pose={request.poseImgUrl}, userId={request.userId}")
    
    print("⏳ Try-On 이미지 생성 중...")
    time.sleep(2)

    file_id = random.randint(10000, 99999)
    base_s3_url = "https://your-s3-bucket.s3.ap-northeast-2.amazonaws.com/avatars"
    tryon_image_url = f"{base_s3_url}/{file_id}_tryon.jpg"

    print(f"✅ Try-On 이미지 생성 완료: {tryon_image_url}")

    return AvatarTryOnResponse(tryOnImgUrl=tryon_image_url)

# --- 서버 실행 ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
