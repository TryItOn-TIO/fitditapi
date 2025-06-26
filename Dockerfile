# CUDA 지원 GPU 환경을 위한 베이스 이미지
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 작업 디렉토리 설정
WORKDIR /app

# 환경변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 시스템 의존성 설치 및 정리
RUN apt-get update && apt-get install -y \
    git wget curl ffmpeg libgl1-mesa-glx \
    python3.10 python3-pip python3.10-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python 기본 명령어 설정
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Pip 업그레이드
RUN python -m pip install --upgrade pip

# Python 의존성 설치 (이미지의 유일한 역할)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 일반 사용자 생성
RUN useradd -ms /bin/bash appuser
USER appuser

# 컨테이너가 종료되지 않고 대기하도록 설정 (개발 시 유용)
CMD ["tail", "-f", "/dev/null"]