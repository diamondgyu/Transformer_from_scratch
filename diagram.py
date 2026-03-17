from diagrams import Cluster, Diagram
from diagrams.aws.ml import Sagemaker
from diagrams.aws.compute import Lambda, ECR
from diagrams.aws.storage import S3
from diagrams.aws.network import APIGateway, CloudFront

with Diagram("Translation ML Pipeline", show=True, direction="LR"):
    with Cluster("Frontend"):
        web_s3 = S3("Web Content (S3)")
        cf = CloudFront("Edge (CloudFront)")

    with Cluster("AWS Cloud"):
        s3 = S3("Model (S3)")
        ecr = ECR("Image (ECR)")
        sm = Sagemaker("SageMaker Inference") 
        lambda_func = Lambda("Inference (Lambda)")
        api = APIGateway("API Gateway")

    web_s3 >> cf
    api >> cf
    [s3, ecr] >> sm
    sm >> lambda_func >> api