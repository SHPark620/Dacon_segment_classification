##  Dacon 신용커드 고객 세그먼트 분류 AI 경전대회

###  문제 개요  
신용커드 고객의 6개월치 시계열 데이터를 기반으로, 각 고객을 A~E 그룹의 세그먼트로 분류하는 문제였습니다.  
총 8가지 영역(회원, 신용, 승인매장 등)에서 수집된 데이터를 기준에, 고객 ID별 시계열 입력을 구성하고  
정보가 된 Segment 레이블을 예측하는 **멀티클래스 분류 문제**입니다.

특히 **B, C 그룹에 해당하는 고객 수가 적어 클래스 분혜률 문제가 매우 심각한 상황**이었으며,  
이 문제를 구현하기 위한 데이터 처리 및 학습 전략 수백이 주요 과제였습니다.

---

###  데이터 구성 및 전체 전처리  

- 6개월(201807~201812) 동안의 고객 데이터를 기준년월, 고객 ID를 기준으로 백화
- 총 65개의 주요 feature를 선별하여 입력 변수로 사용
- 버뮤형 변수는 category encoding, 수치형 변수는 월별 `StandardScaler`로 정규화 적용
- 6개월 모두 존재하는 고객만 filtering 하여 최종 시계열 입력 생성
- **클래스 분혜률 문제**에 대응하기 위해 소수 클래스에 대한 **오버샴플리버팅 적용**

→ 최종 입력 형태: `(Customer, 6 months, 65 features)`

---

###  머들 아키템처: Transformer 기반 분류기

| 구성 요소 | 설명 |
|-----------|------|
| **Input Projection** | `Linear(feat_dim → emb_dim)`  |
| **시계열 인코더** | `nn.TransformerEncoder` (4 layers, 8 heads) |
| **Pooling** | mean pooling (워드 반영 x) |
| **Classification Head** | `Linear(emb_dim → 64) → ReLU → Linear(64 → 5 classes)` |

→ 최종 구성:  
`MLP(Input Embedding)` + `Transformer Encoder` + `Classification Head`

---
###  Project Structure

```
src/                   # 소스코드 train.py=> lstm구조, train_best.py=> 최종 transformer구조
README.md              # 프로젝트 설명 문서
```
---

### ️ 학습 설정  

- **Validation**: `StratifiedKFold(n_splits=5)`  
- **Loss Function**: `CrossEntropyLoss`  
- **Optimizer**: `Adam`, `lr=5e-4`, `weight_decay=1e-4`  
- **Scheduler**: `CosineAnnealingLR`  
- **Mixed Precision**: `torch.amp` 자동 확평률 사용
- **Early Stopping**: `patience=20`

---

###  주요 결과

- Public Score: **27 / 273 (PCC: 0.54057)**  
- Private Score: **90 / 273 (PCC: 0.52843)**  
- Fold별 Best Model 저장 (`model_best_fold*.pt`)  
- 예측 결과는 `submission.csv`로 저장, label은 A~E로 복원

---

###  시각화 결과 예시

- 학습 / 검증 loss curve 시각화
- 예측 분포 출력: 각 클래스별 예측 건수 로그

---

###  느낀 점  
경전대회에서는 클래스 분혜률 문제를 해결하기 위한 **데이터 증가(오버샴플리버팅)** 전략과
Transformer 기반 시계열 모델의 적용 가능성에 대한 가치 많은 검토를 수행할 수 있었습니다.  
개발자가 모든 플레이머를 유지할 수 없는 형태의 데이터에서,  
Transformer의 적용 효과와 입력 검사 / 포멧 가공의 중요성을 다시 한번 찮을 수 있는 가치 있는 경험이었습니다.
