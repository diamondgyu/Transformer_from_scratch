# 초기 버전 (25.02)
## 의의
- DE -> EN (WMT19) 번역, 작동은 잘 함

## 한계점
### 각종 버그
- Attention Scaling 관련 문제
- sacrebleu 를 통한 batch scoring 을 하지 않아 점수 뻥튀기 가능성 존재

### 최적화 미스
- KV cache 미사용
- 인코더와 디코더가 분리되어 있지 않음
- 그것때문에 인코더 값 재사용 불가
- 버퍼 미사용
- bf16 미사용
- validation 을 너무 많이 해서 성능 이슈 존재
- Beam search 만들어두고 쓰지 않음 (내가 짜지 않은 걸 쓰는데 거부감이 있었음)

# 새 버전 만들기
- 토크나이저 교체 (FacebookAI/xlm-roberta-base)
- 다국어 모델(XLM-R)의 과도한 Vocab Size로 인해 가정용 컴퓨터 학습 불가 -> klue/roberta-base로 변경
- lr 이 지나치게 커 그래디언트 폭주 (원인을 정확히 알 수는 없으나 독일어 버전에 비해 바뀐 한국어용 토크나이저 관련 문제일 것으로 추정) -> lr 조정 (scale에 0.5 곱)
- bos, eos, pad 등 특수토큰 매핑 (klue 는 bos, eos 가 아닌 cls sep 임)
- 한국어는 학습 잘 안됨 -> 배치 줄이고 stack을 8로 늘려 학습