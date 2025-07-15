# tasks.py

from celery import Celery
from celery.signals import worker_process_init
from datetime import datetime

# 설정 및 서비스 로직 임포트
from config import settings
# s3_handler.py와 vton_service.py는 프로젝트 내 다른 파일이므로
# 상대 경로 또는 절대 경로 임포트 방식을 프로젝트 구조에 맞게 사용해야 합니다.
# 예: from . import s3_handler, vton_service
import s3_handler
import vton_service

# --- Celery 애플리케이션 생성 ---
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    broker_connection_retry_on_startup=True
)

# --- 워커 프로세스 초기화 시 모델 로드 (워커에서만 실행됨) ---
@worker_process_init.connect
def init_model(**kwargs):
    print("==========================================")
    print("🚀 Celery Worker: Initializing Model...")
    vton_service.generator = vton_service.FitDiTGenerator(settings)
    print("✅ Celery Worker: Model Loaded Successfully!")
    print("==========================================")


# --- Celery 작업(Task) 정의 ---

# --- 👇 여기가 핵심 수정 부분입니다 ---
@celery_app.task(name="process_generate_request")
def process_generate_request(tryOnImgUrl: str, userId: int):
    """
    /generate 요청을 효율적으로 처리하는 작업.
    새로운 vton_service 함수들을 사용하여 중복 연산을 제거합니다.
    """
    print(f"WORKER: Received generate task for userId: {userId}")
    try:
        # 1. S3에서 원본 이미지 다운로드
        model_image = s3_handler.download_image_from_s3(tryOnImgUrl)

        # 2. 포즈 관련 데이터 생성 (시간이 많이 소요되는 작업)
        # vton_service.create_pose_data 함수를 호출합니다.
        pose_image, candidate, vton_img_det = vton_service.create_pose_data(model_image)

        # 3. 생성된 데이터를 재사용하여 상의/하의 마스크 생성 (빠른 작업)
        # vton_service.create_mask_only 함수를 호출합니다.
        upper_mask = vton_service.create_mask_only(model_image, vton_img_det, candidate, "Upper-body")
        lower_mask = vton_service.create_mask_only(model_image, vton_img_det, candidate, "Lower-body")

        # 4. 생성된 이미지들을 S3에 업로드
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        base_key = f"users/{userId}/{timestamp}"
        
        pose_url = s3_handler.upload_pil_image_to_s3(pose_image, f"{base_key}_pose.png")
        upper_mask_url = s3_handler.upload_pil_image_to_s3(upper_mask, f"{base_key}_upper_mask.png")
        lower_mask_url = s3_handler.upload_pil_image_to_s3(lower_mask, f"{base_key}_lower_mask.png")

        print(f"WORKER: Generate task completed efficiently for userId: {userId}")
        return {
            "tryOnImgUrl": tryOnImgUrl,
            "poseImgUrl": pose_url,
            "upperMaskImgUrl": upper_mask_url,
            "lowerMaskImgUrl": lower_mask_url
        }
    except Exception as e:
        print(f"WORKER ERROR (generate): {e}")
        raise e


@celery_app.task(name="process_tryon_request")
def process_tryon_request(request_data: dict):
    """/tryon 요청을 실제로 처리하는 작업"""
    userId = request_data['userId']
    print(f"WORKER: Received tryon task for userId: {userId}")
    try:
        base_image = s3_handler.download_image_from_s3(request_data['baseImgUrl'])
        garment_image = s3_handler.download_image_from_s3(request_data['garmentImgUrl'])
        mask_image = s3_handler.download_image_from_s3(request_data['maskImgUrl'])
        pose_image = s3_handler.download_image_from_s3(request_data['poseImgUrl'])

        result_image = vton_service.perform_try_on(
            base_image=base_image,
            garment_image=garment_image,
            mask_image=mask_image,
            pose_image=pose_image
        )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        result_key = f"results/{userId}/tryon_{timestamp}.png"
        result_url = s3_handler.upload_pil_image_to_s3(result_image, result_key)
        
        print(f"WORKER: Tryon task completed for userId: {userId}")
        return {"tryOnImgUrl": result_url}
        
    except Exception as e:
        print(f"WORKER ERROR (tryon): {e}")
        raise e