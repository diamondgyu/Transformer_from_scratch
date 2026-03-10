from diagrams import Cluster, Diagram
# 출력 리스트에 있는 정확한 명칭 사용: Sagemaker (M 소문자)
from diagrams.aws.ml import Sagemaker, SagemakerModel
from diagrams.aws.compute import Lambda, ECR
from diagrams.aws.storage import S3

with Diagram("Translation ML Pipeline", show=True, direction="LR"):
    with Cluster("AWS Cloud"):
        s3 = S3("Model (S3)")
        ecr = ECR("Image (ECR)")
        # SagemakerModel 아이콘이 더 예쁠 수 있으니 선택해서 쓰세요
        sm = Sagemaker("SageMaker Inference") 
        lambda_func = Lambda("API (Lambda)")

        [s3, ecr] >> sm >> lambda_func