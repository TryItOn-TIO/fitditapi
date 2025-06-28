import boto3
from PIL import Image
import io
from urllib.parse import urlparse
from .config import settings

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_DEFAULT_REGION
)

def get_key_from_url(url: str) -> str:
    """S3 URL에서 객체 키를 추출합니다."""
    parsed_url = urlparse(url)
    # URL 경로의 첫 '/'를 제거하여 키를 얻습니다.
    return parsed_url.path.lstrip('/')

def download_image_from_s3(s3_url: str) -> Image.Image:
    """S3 URL에서 이미지를 다운로드하여 PIL Image 객체로 반환합니다."""
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

def upload_pil_image_to_s3(image: Image.Image, key: str) -> str:
    """PIL Image 객체를 S3에 업로드하고 해당 URL을 반환합니다."""
    try:
        bucket_name = settings.S3_BUCKET_NAME
        buffer = io.BytesIO()
        
        # 이미지 포맷 결정
        file_extension = key.split('.')[-1].lower()
        if file_extension == 'png':
            image_format = 'PNG'
        else: # 기본값은 JPEG
            image_format = 'JPEG'
            
        image.save(buffer, format=image_format)
        buffer.seek(0)
        
        print(f"Uploading to S3: bucket={bucket_name}, key={key}")
        s3_client.upload_fileobj(buffer, bucket_name, key)
        
        # 업로드된 객체의 URL 생성
        url = f"https://{bucket_name}.s3.{settings.AWS_DEFAULT_REGION}.amazonaws.com/{key}"
        print(f"Upload complete. URL: {url}")
        return url
    except Exception as e:
        print(f"S3 업로드 오류: {e}")
        raise
