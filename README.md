# Transformer from Scratch

한국어->영어 번역을 위한 Transformer 모델을 학습하고, AWS Serverless 기반으로 배포해 API 형태로 제공하는 프로젝트
ML 모델 학습과 AWS 배포, Lambda 연동 등 전체 파이프라인을 직접 구현해보는 것을 목적으로 합니다.

# Model Training

## 목표
- 법률/전문 문장을 포함한 한국어 문장을 영어로 번역하는 경량 Transformer(0.1B) 모델 학습
- 단순 동작 수준을 넘어, Beam Search나 KV Cache 등 추론 품질/효율 개선 요소를 반영

## 데이터셋
아래 데이터셋을 조합해 parallel corpus를 구성했습니다.

1. [한국어-영어 번역(병렬) 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=126) (AI Hub)
2. [일상생활 및 구어체 한-영 번역 병렬 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71265) (AI Hub)  
3. [전문분야 한영 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?pageIndex=9&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&srchDataRealmCode=REALM002&aihubDataSe=data&dataSetSn=111) (AI Hub)


## 학습 과정 요약
- 초기 버전(25.03): DE->EN 실험 경험 기반으로 동작은 했지만, attention scaling/sacrebleu 평가/추론 최적화 측면에서 한계가 명확했습니다.
- 개선 버전(26.03):
	- 토크나이저를 `FacebookAI/xlm-roberta-base`에서 `klue/roberta-base`로 변경
	- 한국어 토크나이저 특수토큰(`bos/eos/pad`) 매핑 보강
	- LR 스케일 조정, 헤드/스택 조정 등으로 학습 정체 구간 개선
	- Beam Search, KV Cache, GQA, SwiGLU, RoPE 등 구조적 개선 반영

상세 로그는 [model_training/update_log.md](model_training/update_log.md)에서 확인할 수 있습니다.

## 품질
- Validation BLEU 구간: 약 `0.28 ~ 0.68`
- 전문 분야 도메인 샘플에서도 비교적 안정적인 문장 구조 유지
- 문장 길이가 길거나 동음이의어가 많은 경우 다소 성능이 떨어지나, 주요 의미 전달은 대체로 성공

실제 샘플 일부 분석은 [model_training/sample_generation.md](model_training/sample_generation.md)에 정리되어 있습니다.

# Deployment on AWS

## 아키텍처
현재 배포 경로는 아래와 같습니다.

`S3(model artifacts) + ECR(image) -> SageMaker Endpoint(Serverless Inference) -> Lambda(URL)`

- 모델 파일은 `int8` 양자화된 상태로 S3에 업로드
- S3에 저장된 모델 파일을 SageMaker 경로(`/opt/ml/model`)에 탑재
- 추론 서버 이미지는 ECR에 업로드
- SageMaker Serverless Endpoint가 실제 추론 수행
- Lambda는 외부 요청을 받아 Endpoint를 invoke하는 진입점 역할

## 배포 전략 선택 배경
- EC2 직접 배포: 상시 인스턴스 비용 부담이 큼 (7만원/mo 이상)
- Lambda 단독 모델 구동: 메모리 제약으로 OOM 발생
- 최종 선택: SageMaker Serverless Inference (비용/성능 균형)

## 서버 인터페이스

- `app.py``에서 FastAPI로 서버 구현
- `GET /ping` : 헬스체크
- `POST /invocations` : 추론 호출

요청 예시:

```json
{
	"text": "양도계약의 합의파기에 의한 소유권이전등기의 말소 시 취득처리에 대한 취득세 부과여부에 대한 판경"
}
```

응답 예시:

```json
{
	"translation": "Whether or not the acquisition tax is imposed on the cancellation of the ownership transfer registration based on the settlement of the transfer contract."
}
```

바로 써볼 수 있는 테스트 스크립트 (public endpoint via lambda URL):

```bash
curl -X POST "https://k3tor5wxkjmu66oz3ypvfpvfo40noryg.lambda-url.ap-northeast-2.on.aws/" \
     -H "Content-Type: application/json" \
     -d '{"text": "양도계약의 합의파기에 의한 소유권이전등기의 말소 시 취득처리에 대한 취득세 부과여부에 대한 판결"}'
```

## 이미지 빌드 및 푸시
`aws_deployment` 폴더에서:

```bash
./build_upload_container.sh
```

### 주요 파일들

- [aws_deployment/README.md](aws_deployment/README.md)
- [aws_deployment/build_upload_container.sh](aws_deployment/build_upload_container.sh)
- [aws_deployment/Dockerfile](aws_deployment/Dockerfile)
- [aws_deployment/src/app.py](aws_deployment/src/app.py)
- [aws_deployment/src/sample_request.py](aws_deployment/src/sample_request.py)

# Frontend
