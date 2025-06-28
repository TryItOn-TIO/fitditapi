import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # S3 설정 (환경 변수에서 로드)
    S3_BUCKET_NAME: str = "your-s3-bucket-name"
    AWS_ACCESS_KEY_ID: str = "YOUR_AWS_ACCESS_KEY"
    AWS_SECRET_ACCESS_KEY: str = "YOUR_AWS_SECRET_KEY"
    AWS_DEFAULT_REGION: str = "ap-northeast-2"

    # 모델 및 실행 관련 설정
    MODEL_ROOT: str = "/models"
    MAIN_MODEL_PATH: str = "/models/FitDiT"
    DEVICE: str = "cuda:0"
    WITH_FP16: bool = True
    OFFLOAD: bool = False
    AGGRESSIVE_OFFLOAD: bool = True
    TRYON_RESOLUTION: str = "1152x1536"
    STEPS: int = 20
    GUIDANCE_SCALE: float = 2.0
    SEED: int = -1

    class Config:
        # .env 파일이 있다면 로드 (선택 사항)
        env_file = ".env"
        env_file_encoding = 'utf-8'

# 설정 객체 인스턴스화
settings = Settings()
