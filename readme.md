1. git clone Fitdit
```Bash
git clone https://github.com/BoyuanJiang/FitDiT.git
cd FitDiT
```

2. 모델 다운로드
```Bash
# Git LFS 활성화 (시스템에 따라 최초 한 번만 실행)
sudo apt-get install git-lfs
git lfs install

# Hugging Face 저장소에서 모델 파일을 'local_model_dir' 폴더로 다운로드
git clone https://huggingface.co/BoyuanJiang/FitDiT models
```

3. 입력 이미지 준비 (건너 뛰기)
가상 피팅에 사용할 모델(사람) 이미지와 의상 이미지를 준비합니다. 찾기 쉽도록 images라는 디렉토리를 만들고 그 안에 이미지를 넣어두는 것을 권장합니다.

```Bash
mkdir images
# 예: images/model_person.jpg, images/garment_top.png 와 같이 이미지를 배치합니다.
```

이 부분은 테스트용 application 실행 시 건너 뒴

4. 실행 스크립트 설정 (건너 뛰기)
제공된 fitdit_module.py 파일의 메인 실행 블록(if __name__ == "__main__":)에서 사용할 이미지 경로와 의상 카테고리를 설정합니다.
```python
# fitdit_module.py 파일의 마지막 부분

if __name__ == "__main__":
    # ... (생략) ...

    # --- 사용자가 지정해야 할 부분 ---
    # 3단계에서 준비한 이미지의 경로를 입력합니다. (컨테이너 내부 경로 기준)
    MODEL_IMAGE_PATH = "images/model_person.jpg"
    GARMENT_IMAGE_PATH = "images/garment_top.png"
    OUTPUT_DIR = "results"
    # 의상 카테고리 지정: "Upper-body", "Lower-body", "Dresses" 중 하나 선택
    GARMENT_CATEGORY = "Upper-body"
    # ---------------------------------

    # ... (이하 코드 생략) ...

```

5. 모델 다운로드
```Bash
# 1. 모델들을 담을 부모 폴더 생성
mkdir -p models

# 2. Git-LFS 활성화
git lfs install

# 3. FitDiT 메인 모델 다운로드
git clone https://huggingface.co/BoyuanJiang/FitDiT models/FitDiT

# 4. CLIP-L 모델 다운로드
git clone https://huggingface.co/openai/clip-vit-large-patch14 models/clip-vit-large-patch14

# 5. CLIP-bigG 모델 다운로드
git clone https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k models/CLIP-ViT-bigG-14-laion2B-39B-b160k
```

6. 도커 빌드
```Bash
docker build -t fitdit .
```

7. Docker 컨테이너 실행 (건너 뛰기)
```Bash
# 현재 디렉토리의 절대 경로를 변수에 저장 (Linux/macOS)
HOST_PATH=$(pwd)

# Docker 컨테이너를 실행하여 가상 피팅을 수행합니다.
docker run --gpus all -it --rm \
  -v "${HOST_PATH}/local_model_dir:/app/local_model_dir" \
  -v "${HOST_PATH}/images:/app/images" \
  -v "${HOST_PATH}/results:/app/results" \
  fitdit \
  python fitdit_module.py

```

8. Docker 컨테이너 실행 (FastApi)
# FitDiT_Project 폴더에서 실행
docker run --gpus all -it --rm \
  -v "$(pwd)/FitDiT:/app" \        
  -v "$(pwd)/models:/models" \
  -v "$(pwd)/images:/app/images" \
  -v "$(pwd)/results:/app/results" \
  -w /app \
  fitdit python application