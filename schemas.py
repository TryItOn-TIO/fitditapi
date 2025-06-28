from pydantic import BaseModel, HttpUrl

# --- /generate 엔드포인트 모델 ---

class AvatarCreateRequest(BaseModel):
    tryOnImgUrl: HttpUrl
    userId: int

class AvatarCreateResponse(BaseModel):
    tryOnImgUrl: HttpUrl
    poseImgUrl: HttpUrl
    lowerMaskImgUrl: HttpUrl
    upperMaskImgUrl: HttpUrl

# --- /tryon 엔드포인트 모델 ---

class AvatarTryOnRequest(BaseModel):
    baseImgUrl: HttpUrl
    garmentImgUrl: HttpUrl # 의상 이미지 URL 추가
    maskImgUrl: HttpUrl
    poseImgUrl: HttpUrl
    userId: int

class AvatarTryOnResponse(BaseModel):
    tryOnImgUrl: HttpUrl
