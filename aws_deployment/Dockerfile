FROM python:3.12-slim

# 시스템 의존성 및 권한 설정
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

# 환경 변수 설정 (SageMaker 표준 경로)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/huggingface \
    MODEL_PATH=/opt/ml/model/model-quantized.onnx \
    TOKENIZER_PATH=/opt/ml/model/tokenizer \
    TOKENIZER_MAX_LEN=128 \
    PYTHONPATH="/opt/program"

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# serve 파일에 실행 권한 부여 (반드시 필요)
RUN chmod +x /opt/program/serve

# 환경 변수 PATH에 현재 디렉토리 추가 (SageMaker가 serve를 찾을 수 있게)
ENV PATH="/opt/program:${PATH}"

# 포트 개방
EXPOSE 8080

# 이제 CMD를 "serve"로 바꿉니다.
ENTRYPOINT ["serve"]