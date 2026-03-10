export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=$(aws configure get region)
export ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_URL}
aws ecr create-repository --repository-name my-translator --region ${AWS_REGION} >/dev/null 2>&1 || true

docker rmi my-translator 2>/dev/null || true
docker build --platform linux/amd64 --provenance=false -t my-translator .
docker tag my-translator:latest ${ECR_URL}/my-translator:latest

docker push ${ECR_URL}/my-translator:latest