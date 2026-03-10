## SageMaker 엔드포인트

S3(model), ECR(image) -> SageMaker Endpoint(Serverless) -> Lambda
- 모델과 이미지는 S3와 ECR에 업로드
- SageMaker Endpoint는 Serverless Inference 구성으로 생성
- Lambda는 SageMaker Endpoint를 호출하여 추론 수행

- 모델을 EC2 인스턴스에 배포하는 것도 가능하지만, 비용 과다(7만원/mo)
- Serverless 를 lambda에 배포하는 것도 가능하지만, 성능 부족; 고작 0.1b 모델을 구동하는데도 OOM
- 최종적으로 SageMaker Serverless Inference를 사용하여 배포

### Sample Test
```bash
curl -X POST "https://k3tor5wxkjmu66oz3ypvfpvfo40noryg.lambda-url.ap-northeast-2.on.aws/" \
     -H "Content-Type: application/json" \
     -d '{"text": "양도계약의 합의파기에 의한 소유권이전등기의 말소 시 취득처리 에 대한 취득세 부과여부에 대한 판경"}'
```

응답 예시:
```json
{"translation": "Whether or not the acquisition tax is imposed on the cancellation of the ownership transfer registration based on the settlement of the transfer contract."}
```

- 완전하지는 않지만, 대체로 원문과 유사한 번역 결과
