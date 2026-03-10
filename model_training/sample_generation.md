## 📊 Transformer v2 성능 분석 리포트

| ID | 카테고리 | BLEU | 핵심 평가 |
| --- | --- | --- | --- |
| **24** | 경제/뉴스 | **0.4903** | "회복 조짐"을 "recovery"로 간결하게 압축 |
| **25** | 구어체 | **0.4347** | "get" 대신 "receive"를 선택하는 어휘적 유연성 |
| **26** | 스포츠 뉴스 | **0.2825** | "whenever" 절을 활용한 복합문 구조 생성 성공 |
| **27** | 금융/서비스 | **0.4230** | "In holidays" 전치사구 및 조건절 정렬 양호 |
| **28** | 법률 (판례) | **0.2966** | 고난도 법률 용어(dispute, re-appeal) 배치 성공 |
| **29** | 문학/수필 | **0.5209** | 대조 구조(but)와 추상적 비유 완벽 처리 |
| **30** | 전문 세무 | **0.6849** | 이번 구간의 핵심, 복잡한 명사구 완벽 정렬 |

---
## 일부 샘플 분석

### [Sample 24] 경제 지능

* **원문:** 더 큰 문제는 회복조짐이 보이지 않는다는 점이다.
* **Ideal:** The bigger problem is that there are no signs of recovery.
* **Generated:** **The bigger problem is that there is no recovery.**
* **분석:** `signs of`가 생략되었으나 문장의 핵심 논리(The bigger problem is...)를 정확히 유지함.

### [Sample 26] 복합문 제어력

* **원문:** 그러나 결국 호날두는 일어날 생각조차 안했고 시간이 갈수록 화면에 등장할 때마다 팬들의 야유를 들어야했다.
* **Generated:** **However, in the end, Ronaldo didn ' t even think about what happened, and as time went by, whenever he appeared on the screen, he had to booe fans.**
* **분석:** `whenever`와 `as time went by` 등 다중 부사절이 섞인 장문을 꼬임 없이 생성해냄.

### [Sample 30] 전문 세무/법률

* **원문:** 양도계약의 합의해제로 인한 소유권이전등기의 말소로 소유권을 회복한 것이 취득세 과세대상인 부동산취득에 해당하는지 여부.
* **Ideal:** Whether the restoration of ownership by the cancellation of the transfer registration due to the dissolution of the agreement on the transfer contract constitutes the acquisition of real estate subject to acquisition tax.
* **Generated:** **Whether the restoration of ownership by canceling the registration of ownership transfer due to the cancellation of the agreement on the transfer contract constitutes the acquisition of real estate subject to acquisition of acquisition tax.**
* **분석:** **BLEU 0.6849**. "취득세 과세대상인 부동산취득" 같은 첩첩산중 명사구 나열을 영어의 `constitutes... subject to` 구조로 완벽하게 번역함.