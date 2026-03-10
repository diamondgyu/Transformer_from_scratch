from transformers import AutoTokenizer

# 사용 중인 모델명 입력
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
tokenizer.save_pretrained("./models/tokenizer")