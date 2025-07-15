# s3_handler.py

import boto3
from PIL import Image
import io
from urllib.parse import urlparse
from config import settings

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_DEFAULT_REGION
)

# S3 URL에서 객체 키를 추출하는 헬퍼 함수
def get_key_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    return parsed_url.path.lstrip('/')

# --- 👇 여기가 가장 중요한 수정 부분입니다 (async 삭제) ---
def download_image_from_s3(s3_url: str) -> Image.Image:
    """S3 URL에서 이미지를 동기적으로 다운로드하여 PIL Image 객체로 반환합니다."""
    try:
        bucket_name = settings.S3_BUCKET_NAME
        key = get_key_from_url(s3_url)
        
        print(f"Downloading from S3: bucket={bucket_name}, key={key}")
        
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print("Download complete.")
        return image
    except Exception as e:
        print(f"S3 다운로드 오류: {e}")
        raise

# --- 👇 여기도 마찬가지로 async 삭제 ---
def upload_pil_image_to_s3(image: Image.Image, key: str) -> str:
    """PIL Image 객체를 S3에 동기적으로 업로드하고 해당 URL을 반환합니다."""
    try:
        bucket_name = settings.S3_BUCKET_NAME
        buffer = io.BytesIO()
        
        file_extension = key.split('.')[-1].lower()
        image_format = 'PNG' if file_extension == 'png' else 'JPEG'
            
        image.save(buffer, format=image_format)
        buffer.seek(0)
        
        print(f"Uploading to S3: bucket={bucket_name}, key={key}")
        s3_client.upload_fileobj(buffer, bucket_name, key)
        
        url = f"https://{bucket_name}.s3.{settings.AWS_DEFAULT_REGION}.amazonaws.com/{key}"
        print(f"Upload complete. URL: {url}")
        return url
    except Exception as e:
        print(f"S3 업로드 오류: {e}")
        raise