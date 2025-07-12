import boto3
from PIL import Image
import io
from urllib.parse import urlparse
import asyncio  # asyncio 라이브러리를 임포트합니다.
from config import settings

# boto3의 동기 클라이언트를 생성합니다.
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_DEFAULT_REGION
)

def get_key_from_url(url: str) -> str:
    """S3 URL에서 객체 키를 추출합니다."""
    parsed_url = urlparse(url)
    return parsed_url.path.lstrip('/')

# --- 동기적으로 실행될 실제 작업 함수들 ---

def _download_image_sync(s3_url: str) -> Image.Image:
    """S3에서 이미지를 동기적으로 다운로드합니다. (실제 작업)"""
    bucket_name = settings.S3_BUCKET_NAME
    key = get_key_from_url(s3_url)
    print(f"Sync download starting in background thread for key: {key}")
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    image_data = response['Body'].read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    print(f"Sync download complete for key: {key}")
    return image

def _upload_image_sync(image: Image.Image, key: str) -> str:
    """PIL 이미지를 S3에 동기적으로 업로드합니다. (실제 작업)"""
    bucket_name = settings.S3_BUCKET_NAME
    buffer = io.BytesIO()
    image_format = 'PNG' if key.endswith('.png') else 'JPEG'
    image.save(buffer, format=image_format)
    buffer.seek(0)
    
    print(f"Sync upload starting in background thread for key: {key}")
    s3_client.upload_fileobj(buffer, bucket_name, key)
    
    url = f"https://{bucket_name}.s3.{settings.AWS_DEFAULT_REGION}.amazonaws.com/{key}"
    print(f"Sync upload complete. URL: {url}")
    return url

# --- 비동기 인터페이스 함수들 ---

async def download_image_from_s3(s3_url: str) -> Image.Image:
    """동기 다운로드 함수를 별도 스레드에서 실행하여 비동기처럼 동작하게 만듭니다."""
    return await asyncio.to_thread(_download_image_sync, s3_url)

async def upload_pil_image_to_s3(image: Image.Image, key: str) -> str:
    """동기 업로드 함수를 별도 스레드에서 실행하여 비동기처럼 동작하게 만듭니다."""
    return await asyncio.to_thread(_upload_image_sync, image, key)