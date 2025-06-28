FitDiT 기반 가상 피팅(Virtual Try-on) API이 프로젝트는 FitDiT 모델을 사용하여 상의, 하의 가상 피팅 기능을 제공하는 고성능 FastAPI 서버입니다. Docker를 통해 모든 의존성을 관리하며, AWS S3와 연동하여 이미지 데이터를 효율적으로 처리합니다.✨ 주요 기능모듈식 구조: FastAPI(웹), vton_service(ML), s3_handler(클라우드) 등 기능별로 코드가 분리되어 유지보수가 용이합니다.모델 캐싱: 서버 시작 시 단 한 번만 무거운 ML 모델을 로드하여, 이후 모든 요청을 빠르게 처리합니다.S3 연동: 모든 이미지(원본, 마스크, 포즈, 결과)를 S3에서 관리하여 확장성을 확보합니다.상/하의 마스크 동시 생성: /generate 엔드포인트 호출 한 번으로 상의와 하의 마스크를 모두 생성하여 효율성을 높입니다.📋 사전 준비 사항NVIDIA GPU가 장착된 호스트 머신최신 NVIDIA 드라이버 및 Docker, NVIDIA Container ToolkitAWS 계정 및 이미지 저장을 위한 S3 버킷AWS CLI 또는 환경 변수를 통한 AWS 자격 증명 설정🚀 실행 방법1. 환경 변수 설정프로젝트 실행에 필요한 환경 변수를 설정합니다. 터미널에서 아래 명령어를 실행하거나 .env 파일을 사용하세요.export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
export AWS_DEFAULT_REGION="ap-northeast-2" # 실제 사용하는 리전
export S3_BUCKET_NAME="your-s3-bucket-name" # 생성한 S3 버킷 이름
2. Docker 이미지 빌드프로젝트 루트 디렉토리에서 아래 명령어를 실행하여 Docker 이미지를 빌드합니다.docker build -t fitdit-api .
3. Docker 컨테이너 실행빌드된 이미지를 사용하여 컨테이너를 실행합니다. 로컬의 모델과 이미지를 컨테이너 내부로 마운트하고, 환경 변수를 전달합니다.docker run --gpus all --rm -p 8000:8000 \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/images":/app/images \
  -v "$(pwd)/results":/app/results \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  -e S3_BUCKET_NAME \
  fitdit-api
-v "$(pwd)/models":/models: 호스트의 모델 폴더를 컨테이너의 /models로 연결합니다.-v "$(pwd)/images":/app/images: 테스트용 이미지를 컨테이너 내부로 연결합니다.-v "$(pwd)/results":/app/results: 결과물을 호스트에서 확인하기 위해 연결합니다.-e ...: 호스트의 환경 변수를 컨테이너로 전달합니다.🧪 API 테스트 (cURL 예시)S3에 source/model.png 와 source/shirt.png, source/pants.png를 미리 업로드하세요.1. 마스크 및 포즈 생성 (/generate)curl -X POST http://localhost:8000/generate \
-H "Content-Type: application/json" \
-d '{
    "tryOnImgUrl": "https://your-s3-bucket-name.s3.ap-northeast-2.amazonaws.com/source/model.png",
    "userId": 123
}'
2. 상의 Try-On (/tryon)# /generate에서 받은 응답값을 사용합니다.
curl -X POST http://localhost:8000/tryon \
-H "Content-Type: application/json" \
-d '{
    "baseImgUrl": "https://.../source/model.png",
    "garmentImgUrl": "https://.../source/shirt.png",
    "maskImgUrl": "https://.../masks/123_upper_mask_....png",
    "poseImgUrl": "https://.../poses/123_pose_....png",
    "userId": 123
}'
3. 하의 Try-On (연속)상의 Try-On 결과 이미지 URL을 baseImgUrl로 사용하여 하의를 입힐 수 있습니다.curl -X POST http://localhost:8000/tryon \
-H "Content-Type: application/json" \
-d '{
    "baseImgUrl": "https://.../results/123_tryon_....png", # 상의 Try-On 결과 URL
    "garmentImgUrl": "https://.../source/pants.png",
    "maskImgUrl": "https://.../masks/123_lower_mask_....png", # /generate에서 받은 하의 마스크 URL
    "poseImgUrl": "https://.../poses/123_pose_....png",
    "userId": 123
}'
