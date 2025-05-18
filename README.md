# Dacon 신용카드 고객 세그먼트 분류 AI 경진대회

## 문제 개요  
신용카드 고객의 6개월치 시계열 데이터를 기반으로, 각 고객을 A~E 그룹의 세그먼트로 분류하는 문제였습니다.  
총 8가지 영역(회원, 신용, 승인매장 등)에서 수집된 데이터를 고객 ID 및 기준년월을 기준으로 병합한 후, 결과에 대한 Segment 레이블을 예측하는 **멀티클래스 분류 문제**입니다.

특히 **B, C 그룹에 해당하는 고객 수가 매우 적어 클래스 불균형 문제가 심각한 상황**이었으며, 이 문제를 해결하기 위한 데이터 처리 및 학습 전략 설계가 주요 과제였습니다.

---

## 데이터 구성 및 전처리

- 6개월(201807~201812) 동안의 고객 데이터를 기준년월과 고객 ID 기준으로 병합  
- 총 65개의 주요 feature를 선별하여 입력 변수로 사용  
- 범주형 변수는 category encoding, 수치형 변수는 월별 다른 `StandardScaler`로 정규화 적용  
- 6개월치 데이터가 모두 존재하는 고객만 filtering 하여 최종 시계열 입력 구성  
- **클래스 불균형 문제**에 대응하기 위해 소수 클래스에 대한 **오버샘플링 적용**

→ 최종 입력 형태: `(Customer, 6 months, 65 features)`

---

## 모델 아키텍처: Transformer 기반 분류기

| 구성 요소 | 설명 |
|-----------|------|
| **Input Projection** | `Linear(feat_dim → emb_dim)` |
| **시계열 인코더** | `nn.TransformerEncoder` (4 layers, 8 heads) |
| **Pooling** | mean pooling (Time dimension 평균) |
| **Classification Head** | `Linear(emb_dim → 64) → ReLU → Linear(64 → 5 classes)` |

→ 최종 구성:  
`MLP(Input Embedding)` + `Transformer Encoder` + `Classification Head`

---

## Project Structure

```
src/                   # 소스코드 디렉토리
  ├─ train.py          # LSTM 기반 실험용 모델
  └─ train_best.py     # 최종 Transformer 기반 구조
README.md              # 프로젝트 설명 문서
```

---

## 학습 설정  

- **Validation**: `StratifiedKFold(n_splits=5)`  
- **Loss Function**: `CrossEntropyLoss`  
- **Optimizer**: `Adam`, `lr=5e-4`, `weight_decay=1e-4`  
- **Scheduler**: `CosineAnnealingLR`  
- **Mixed Precision**: `torch.amp`  
- **Early Stopping**: `patience=20`

---

## 느낀 점  

이번 경진대회에서는 클래스 불균형 문제를 해결하기 위한 **데이터 증가(오버샘플링)** 전략과  
**Transformer 기반 시계열 모델의 적용 가능성**에 대해 깊이 있게 검토해볼 수 있었습니다.  
특히, **개발자가 모든 feature를 명확히 통제할 수 없는 복잡한 형태의 데이터** 환경에서  
Transformer 모델이 실제로 어떻게 동작하는지를 실험하며,  
**입력 데이터의 전처리 및 포맷 가공이 모델 성능에 매우 큰 영향을 미친다**는 것을 다시금 **실감할 수 있었던 값진 경험**이었습니다.
