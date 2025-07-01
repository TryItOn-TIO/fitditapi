#!/bin/bash

# 스크립트 실행 중 오류가 발생하면 즉시 중단합니다.
set -e

# --- 스크립트 설명 ---
# 이 스크립트는 FitDiT 프로젝트의 설치 및 실행을 자동화합니다.
# 1. 프로젝트 폴더 생성 및 이동
# 2. FitDiT GitHub 저장소 복제
# 3. Hugging Face에서 필요한 AI 모델 다운로드 (git-lfs 필요)
# 4. Docker 이미지 빌드
# 5. 모든 볼륨을 마운트하여 Docker 컨테이너 실행

# --- 메인 스크립트 ---

echo "FitDiT 자동 설치 및 실행을 시작합니다."
echo "=========================================="
echo

# 1. Git 및 Docker 설치 여부 확인
if ! command -v git &> /dev/null
then
    echo "오류: 'git'이 설치되어 있지 않습니다. Git을 먼저 설치해주세요."
    exit 1
fi

if ! command -v docker &> /dev/null
then
    echo "오류: 'docker'가 설치되어 있지 않습니다. Docker를 먼저 설치해주세요."
    exit 1
fi

echo "1. 프로젝트 루트 폴더를 생성합니다..."
# 최상위 프로젝트 폴더를 생성하고 해당 폴더로 이동합니다.
# 이 폴더 안에 모든 관련 파일(소스코드, 모델 등)이 저장됩니다.
mkdir -p FitDiT_Project
cd FitDiT_Project
echo "작업 디렉토리: $(pwd)"
echo

# 2. FitDiT 메인 저장소를 복제합니다.
echo "2. FitDiT 소스 코드를 다운로드합니다..."
if [ -d "FitDiT" ]; then
    echo "-> 'FitDiT' 폴더가 이미 존재하므로 이 단계를 건너뜁니다."
else
    git clone https://github.com/BoyuanJiang/FitDiT.git
fi
echo

# 3. 필요한 AI 모델들을 다운로드합니다.
echo "3. AI 모델을 다운로드합니다 (용량이 크므로 시간이 걸릴 수 있습니다)..."
mkdir -p models
git lfs install > /dev/null 2>&1

# 3-1. FitDiT 메인 모델
echo " -> FitDiT 메인 모델 다운로드 중..."
if [ -d "models/FitDiT" ]; then
    echo "    -> 모델이 이미 존재하므로 건너뜁니다."
else
    git clone https://huggingface.co/BoyuanJiang/FitDiT models/FitDiT
fi

# 3-2. CLIP-L 모델
echo " -> CLIP-L 모델 다운로드 중..."
if [ -d "models/clip-vit-large-patch14" ]; then
    echo "    -> 모델이 이미 존재하므로 건너뜁니다."
else
    git clone https://huggingface.co/openai/clip-vit-large-patch14 models/clip-vit-large-patch14
fi

# 3-3. CLIP-bigG 모델
echo " -> CLIP-bigG 모델 다운로드 중..."
if [ -d "models/CLIP-ViT-bigG-14-laion2B-39B-b160k" ]; then
    echo "    -> 모델이 이미 존재하므로 건너뜁니다."
else
    git clone https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k models/CLIP-ViT-bigG-14-laion2B-39B-b160k
fi
echo "모든 모델 다운로드가 완료되었습니다."
echo

# 4. Docker 빌드에 필요한 폴더들을 생성합니다.
echo "4. 이미지 입출력을 위한 'images', 'results' 폴더를 생성합니다..."
mkdir -p images
mkdir -p results
echo "-> 폴더 준비 완료."
echo

# 5. Docker 이미지를 빌드합니다.
echo "5. Docker 이미지를 빌드합니다 ('fitdit' 이름으로 생성)..."
# Dockerfile이 있는 FitDiT 소스 코드 폴더로 이동하여 빌드합니다.
cd FitDiT
docker build -t fitdit .
cd .. # 다시 프로젝트 루트 폴더로 돌아옵니다.
echo "-> Docker 이미지 빌드 완료."
echo

# 6. Docker 컨테이너를 실행합니다.
echo "6. Docker 컨테이너를 실행하여 FastAPI 애플리케이션을 시작합니다..."
echo "   - GPU 사용 (--gpus all)"
echo "   - 컨테이너 종료 시 자동 삭제 (--rm)"
echo "   - 소스코드, 모델, 이미지/결과 폴더를 컨테이너와 연결"
echo
echo "애플리케이션을 시작합니다. 중지하려면 'Ctrl + C'를 누르세요."
echo "--------------------------------------------------------"

# 사용자가 제공한 docker run 명령어를 실행합니다.
# $(pwd)를 사용하여 현재 경로(FitDiT_Project)를 기준으로 볼륨을 마운트합니다.
docker run --gpus all -it --rm \
  -v "$(pwd)/FitDiT:/app" \
  -v "$(pwd)/models:/models" \
  -v "$(pwd)/images:/app/images" \
  -v "$(pwd)/results:/app/results" \
  -w /app \
  fitdit python application.py

echo "--------------------------------------------------------"
echo "FitDiT 애플리케이션이 종료되었습니다. 스크립트를 마칩니다."
