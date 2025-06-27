# main.py

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import time
import random

# --- Pydantic 모델 정의 (데이터 유효성 검사 및 타입 힌팅) ---

# 요청 모델: tryOnImgUrl과 userId를 받도록 정의
class AvatarCreateRequest(BaseModel):
    tryOnImgUrl: HttpUrl
    userId: int

# 응답 모델: 4개의 URL 필드를 가지도록 정의
class AvatarCreateResponse(BaseModel):
    tryOnImgUrl: HttpUrl  # 필드명을 tryOnImgUrlL -> tryOnImgUrl 로 수정 (일관성)
    poseImgUrl: HttpUrl
    lowerMaskImgUrl: HttpUrl
    upperMaskImgUrl: HttpUrl

# --- FastAPI 애플리케이션 생성 ---

app = FastAPI(
    title="아바타 이미지 생성 API",
    description="원본 이미지를 받아 Pose와 Mask 이미지를 생성하고 S3 주소를 반환합니다.",
    version="1.0.0"
)

# --- API 엔드포인트 구현 ---

@app.post(
    "/generate",
    response_model=AvatarCreateResponse,
    summary="Pose 및 Mask 이미지 생성",
    description="원본 이미지 S3 주소를 받아, Pose와 Mask 이미지를 생성하고 각각의 S3 주소를 반환합니다."
)
async def generate_images(request: AvatarCreateRequest):
    """
    이 함수는 Spring Boot 서버로부터 아바타 생성 요청을 처리합니다.

    Args:
        request (AvatarCreateRequest): 원본 이미지의 S3 URL과 사용자 ID를 포함하는 요청 바디

    Returns:
        AvatarCreateResponse: 생성된 Pose 및 Mask 이미지의 S3 URL을 포함하는 응답
    """
    # --- 문제 수정 1: 올바른 필드명(tryOnImgUrl) 사용 ---
    print(f"✅ 요청 수신: TryOn 이미지 URL = {request.tryOnImgUrl}, 사용자 ID = {request.userId}")

    # --- 이미지 처리 및 S3 업로드 시뮬레이션 ---
    print("⏳ 1. 원본 이미지 다운로드 중...")
    time.sleep(1)
    print("⏳ 2. Pose 및 Mask 이미지 생성 중...")
    time.sleep(2)
    print("⏳ 3. 생성된 이미지를 S3에 업로드 중...")
    time.sleep(1)

    # 4. S3로부터 받은 URL 생성 (시뮬레이션)
    file_id = random.randint(10000, 99999)
    base_s3_url = "https://your-s3-bucket.s3.ap-northeast-2.amazonaws.com/avatars"
    
    pose_image_s3_url = f"{base_s3_url}/{file_id}_pose.jpg"
    upper_mask_image_s3_url = f"{base_s3_url}/{file_id}_upper_mask.png"
    lower_mask_image_s3_url = f"{base_s3_url}/{file_id}_lower_mask.png"
    
    print(f"✅ 생성 완료: Pose 이미지 URL = {pose_image_s3_url}")
    print(f"✅ 생성 완료: Upper Mask 이미지 URL = {upper_mask_image_s3_url}")
    print(f"✅ 생성 완료: Lower Mask 이미지 URL = {lower_mask_image_s3_url}")

    # --- 문제 수정 2: 응답 모델의 모든 필드를 정확하게 채워서 반환 ---
    return AvatarCreateResponse(
        tryOnImgUrl=request.tryOnImgUrl, # 요청받은 원본 URL을 그대로 반환
        poseImgUrl=pose_image_s3_url,
        upperMaskImgUrl=upper_mask_image_s3_url,
        lowerMaskImgUrl=lower_mask_image_s3_url
    )

# --- 서버 실행 (개발 환경) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
