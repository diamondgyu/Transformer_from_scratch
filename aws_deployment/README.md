## SageMaker Serverless Inference

이 디렉토리는 SageMaker 커스텀 추론 컨테이너 규격으로 동작합니다.

- 헬스체크: `GET /ping`
- 추론: `POST /invocations`

요청 예시:

```json
{
	"text": "번역할 문장"
}
```

응답 예시:

```json
{
	"translation": "Text to be translated"
}
```

## 로컬 실행

```bash
docker build -t my-translator .
docker run --rm -p 8080:8080 my-translator
```

별도 터미널에서:

```bash
cd src
python sample_request.py
```

## ECR 업로드

```bash
./build_upload_docker.sh
```

## SageMaker Endpoint 생성 시 참고

- 이미지 URI: `${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/my-translator:latest`
- 엔드포인트는 Serverless Inference 구성 사용
- 컨테이너는 포트 `8080`에서 `/ping`, `/invocations`를 제공
