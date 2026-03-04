# Transformer 개선 사항 및 최적화 제안 (for vibe coding)

### 1. 아키텍처 및 추론(Inference) 레벨의 결함

* **문제 1: 인코더 중복 연산**
* **현상:** `generate_translation`에서 단어를 하나 생성할 때마다 `self(encoder_input, beam_sequence)`를 호출하여 전체 `TransformerModel`을 통과시킵니다.
* **결과:** 이미 연산이 끝난 입력 문장(Source)의 인코딩을 매 토큰마다 처음부터 다시 계산합니다.
* **개선:** 모델의 `forward` 로직을 분리해야 합니다. 추론 시에는 `encoder_output = self.encode(src)`를 단 한 번만 실행하고, 생성 루프 안에서는 `self.decode(tgt, encoder_output)`만 반복 호출하도록 구조를 개편해야 합니다.

* **문제 2: KV 캐시(Key-Value Cache) 미구현**
* **현상:** `MultiheadAttentionCustom`에 캐시 변수만 선언되어 있고, 실제 `forward`에서는 매번 `query.size(1)`(현재까지 생성된 전체 길이)만큼 Q, K, V 행렬 곱을 전부 다시 수행합니다.
* **개선:** 디코더의 Self-Attention에서 이전 스텝까지 계산된 K, V 텐서를 메모리에 들고 있다가, 새로 들어온 1개의 토큰(Q)과만 연산하도록 캐시 업데이트 로직을 추가해야 합니다.

* **문제 3: 루프 기반의 순차적 Beam Search**
* **현상:** 빔 후보군을 `for beam_idx, ... in enumerate(current_beams)`로 하나씩 모델에 넣습니다.
* **개선:** GPU는 행렬 병렬 연산에 특화되어 있습니다. 여러 개의 빔 시퀀스를 `(batch_size * num_beams, seq_len)` 형태의 단일 텐서로 묶어 한 번에 모델을 통과시키는 배치화(Vectorization)가 필수적입니다.


### 2. 메모리 및 텐서 연산의 비효율성

GPU와 CPU 간의 불필요한 데이터 이동과 메모리 재할당이 빈번하게 일어납니다.

* **문제 1: 매 스텝 발생하는 텐서 할당과 전송**
* **현상:** `tgt_input = torch.zeros_like(tgt).to(device)`처럼 매번 0으로 채운 텐서를 새로 만들고 GPU로 보냅니다. 또한 `pad_mask`와 `target_mask`도 `forward`가 호출될 때마다 CPU에서 생성되어 GPU로 복사됩니다.
* **개선:** * 마스크나 위치 인코딩(Positional Encoding) 같이 고정된 값은 `__init__`에서 `self.register_buffer`로 등록하여 모델과 함께 GPU 메모리에 영구 상주시켜야 합니다.
* `tgt_input`은 GPU 내에서 `torch.full`과 `torch.cat`을 사용하여 메모리 복사 없이 즉시 결합해야 합니다.


### 3. 학습 루프 및 평가(Metric)의 오류

학습 파이프라인의 병목과 부정확한 성능 측정을 유발합니다.

* **문제 1: 과도한 Validation 루프**
* **현상:** 훈련 중 일정 주기마다 전체 Validation Set을 순회합니다. 이 동안 학습은 완전히 멈춥니다.
* **개선:** (이미 50개 배치 제한으로 수정하셨듯) 훈련 중에는 소수의 샘플링된 배치만 확인하고, 전체 평가는 1 에폭이 끝난 후에만 수행해야 합니다.


* **문제 2: Sentence BLEU 평균의 함정**
* **현상:** NLTK를 사용해 개별 문장의 BLEU 점수를 구한 뒤 평균을 냅니다. 짧은 문장에 가중치가 과도하게 부여되어 학술적으로 인정받지 못하는 점수가 나옵니다.
* **개선:** `sacrebleu` 라이브러리를 도입하여, 에폭(Epoch) 단위로 생성된 전체 텍스트 코퍼스(Corpus)에 대해 한 번에 BLEU를 계산해야 객관적인 성능 지표를 얻을 수 있습니다.


# Improvements

### `model.py` — Architecture & Inference

| 개선 사항 | 변경 내용 |
| --- | --- |
| **인코더 중복 연산 제거** | `encode(src)` / `decode(tgt, enc_out)` 분리 → 추론 시 인코더 1회만 실행 |
| **KV 캐시** | `MultiheadAttentionCustom`에 self-attention 캐시 (append 방식) + cross-attention 캐시 (1회 계산 후 재사용) 구현 |
| **벡터화된 Beam Search** | 모든 빔을 `(batch×beam, seq)` 단일 텐서로 묶어 GPU 병렬 처리, `_reorder_cache()`로 캐시 재배치 |
| **register_buffer** | Positional encoding & causal mask를 버퍼로 등록 → `.to(device)` 시 자동 이동, 매 `forward`마다 재생성/전송 제거 |
| **Attention 스케일링 수정** | `√embed_dim` → `√head_dim` (논문 원본 `√d_k` 기준 반영) |
| **LayerNorm 수정** | Encoder: `layer_norm_ffnn` 정상 적용 <br>

<br> Decoder: 3개 서브레이어에 각각 전용 LayerNorm 적용 |
| **Attention dropout** | Attention weights에 dropout 적용 (논문 spec 일치) |
| **bf16 저장/로드** | `save_bf16()` / `load_bf16()` 정적 메서드 구현 (파일 크기 ~50% 감소) |
| **Decoder 임베딩 스케일링** | Encoder와 동일하게 `√embed_dim` 스케일 적용 (기존 누락분 반영) |

---

### `main.py` — Training

| 개선 사항 | 변경 내용 |
| --- | --- |
| **bf16 저장** | `torch.save()` → `TransformerModel.save_bf16()`로 대체 |
| **tgt_input GPU 직접 생성** | `zeros_like().to(device)` → `torch.full()` + `torch.cat()` on GPU (불필요한 CPU↔GPU 전송 오버헤드 제거) |
| **autocast device_type** | `'cuda:0'` → `'cuda'` (올바른 PyTorch API 규약 사용) |
| **Validation 변수 분리** | 훈련용 `src`/`tgt`와 검증용 `v_src`/`v_tgt` 변수명 분리 |
| **test_model avg_loss** | `len(test)` → `count` (실제 처리한 배치 수 기준으로 나누어 정확도 확보) |

---

### `test.py` — Evaluation

| 개선 사항 | 변경 내용 |
| --- | --- |
| **인코더 1회 실행** | `generate()` → `model.greedy_generate()` 호출 구조로 변경 (Encode once + KV-cached decode) |
| **sacrebleu 도입** | NLTK `sentence_bleu` → `sacrebleu.sentence_bleu`로 교체 및 `calculate_corpus_bleu()` (코퍼스 레벨 BLEU) 함수 신설 |
| **strict=False 로딩** | 기존 체크포인트의 `register_buffer` 키 호환을 위한 로딩 옵션 추가 |
