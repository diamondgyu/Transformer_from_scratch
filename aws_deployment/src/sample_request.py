import boto3 # type: ignore
import json
import time

# [필수 수정] AWS 콘솔에서 생성된 SageMaker 엔드포인트 이름으로 변경하세요.
# 예: 'my-translator-endpoint'
ENDPOINT_NAME = "your-sagemaker-endpoint-name"

def invoke_sagemaker_endpoint():
    print(f"🚀 SageMaker 엔드포인트 '{ENDPOINT_NAME}' 호출 준비 중...")
    
    # boto3 클라이언트 생성 (리전은 서울로 고정)
    client = boto3.client('sagemaker-runtime', region_name='ap-northeast-2')
    
    # 모델에 던질 테스트 데이터
    payload = {
        "text": "양도계약의 합의파기시 소유권이전등기의 등록말소로 취득처리시 취득세부과의 여부에 대한 판결"
    }
    
    start_time = time.time()
    
    try:
        # SageMaker 추론 요청 
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json', # FastAPI의 Pydantic 모델 검증을 통과하기 위해 필수
            Body=json.dumps(payload)
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        # StreamingBody로 들어온 결과를 읽고 JSON으로 파싱
        result_body = response['Body'].read().decode('utf-8')
        result_json = json.loads(result_body)
        
        print(f"✅ 호출 성공! (소요 시간: {latency:.2f}초)")
        print("✨ 번역 결과:")
        print(json.dumps(result_json, indent=4, ensure_ascii=False))
        
    except client.exceptions.ValidationError as e:
        print(f"❌ 입력값 검증 에러 (포맷 확인 필요): {e}")
    except client.exceptions.ModelError as e:
        print(f"❌ 모델 내부 에러 (컨테이너 로그 확인 필요): {e}")
    except Exception as e:
        print(f"🚨 연결 또는 알 수 없는 에러: {e}")

if __name__ == "__main__":
    invoke_sagemaker_endpoint()