import os, time, numpy as np, pandas as pd, torch, joblib
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

BASE_TRAIN = "E:/AI_proj/dacon_segment/open/train"
BASE_TEST = "E:/AI_proj/dacon_segment/open/test"
DIRS = [
    "1.회원정보", "2.신용정보", "3.승인매출정보", "4.청구입금정보",
    "5.잔액정보", "6.채널정보", "7.마케팅정보", "8.성과정보"
]
MONTHS = ["201807", "201808", "201809", "201810", "201811", "201812"]

FILENAME_MAP = {
    "1.회원정보": "회원정보",
    "2.신용정보": "신용정보",
    "3.승인매출정보": "승인매출정보",
    "4.청구입금정보": "청구정보",
    "5.잔액정보": "잔액정보",
    "6.채널정보": "채널정보",
    "7.마케팅정보": "마케팅정보",
    "8.성과정보": "성과정보"
}

FEATS = [
    '남녀구분코드', '연령',
    '회원여부_이용가능', '회원여부_이용가능_CA', '회원여부_이용가능_카드론',
    '소지여부_신용', '소지카드수_유효_신용', '소지카드수_이용가능_신용', '입회경과개월수_신용',
    '동의여부_한도증액안내', '수신거부여부_TM', '수신거부여부_DM', '수신거부여부_메일', '수신거부여부_SMS',
    '탈회횟수_누적', '최종탈회후경과월', '마케팅동의여부',
    '유효카드수_신용', '유효카드수_체크', '이용가능카드수_신용', '이용가능카드수_체크',
    '이용카드수_신용', '이용카드수_체크', '이용금액_R3M_신용', '이용금액_R3M_체크',
    '_1순위카드이용금액', '_1순위카드이용건수', '_1순위신용체크구분',
    '_2순위카드이용금액', '_2순위카드이용건수', '_2순위신용체크구분',
    '최종카드발급일자',
    '보유여부_해외겸용_본인', '이용가능여부_해외겸용_본인', '이용여부_3M_해외겸용_본인',
    '보유여부_해외겸용_신용_본인', '이용가능여부_해외겸용_신용_본인', '이용여부_3M_해외겸용_신용_본인',
    '기본연회비_B0M', '제휴연회비_B0M',
    '할인금액_기본연회비_B0M', '할인금액_제휴연회비_B0M',
    '청구금액_기본연회비_B0M', '청구금액_제휴연회비_B0M',
    '카드신청건수', '최종카드발급경과월',
    '홈페이지_금융건수_R3M',
    'IB문의건수_결제_R6M', 'IB문의건수_결제일변경_R6M',
    'IB문의건수_CA_R6M', 'IB문의건수_APP_B0M', 'IB상담건수_VOC민원_R6M',
    '방문일수_PC_B0M'
]

SEGMENT_CACHE = {}
scaler = GradScaler()

def print_diagnostics(X, y, verbose=False):
    if not verbose:
        return
    print("[진단] X contains NaN:", torch.isnan(X).any().item())
    print("[진단] X contains inf:", torch.isinf(X).any().item())
    print("[진단] y contains NaN:", torch.isnan(y.float()).any().item())
    print("[진단] y unique:", y.unique())

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# 오버샘플링 적용 함수
def oversample(X, y, oversample_ratio={0: 100, 1: 200}):
    X_aug, y_aug = [], []
    for cls, ratio in oversample_ratio.items():
        mask = (y == cls)
        X_cls = X[mask]
        y_cls = y[mask]
        for _ in range(ratio):
            X_aug.append(X_cls)
            y_aug.append(y_cls)
    if X_aug:
        X_oversampled = torch.cat([X] + X_aug)
        y_oversampled = torch.cat([y] + y_aug)
        return X_oversampled, y_oversampled
    else:
        return X, y

def load_monthly(base_path, mode="train"):
    monthly = {}
    for m in MONTHS:
        dfs = []
        for d in DIRS:
            name = FILENAME_MAP[d]
            fname = f"{m}_{mode}_{name}.parquet"
            fp = os.path.join(base_path, d, fname)
            if not os.path.exists(fp):
                raise FileNotFoundError(f" 파일 없음: {fp}")
            t = pd.read_parquet(fp)
            t["기준년월"] = m
            if mode == "train" and d == "1.회원정보" and m == MONTHS[-1] and "Segment" in t.columns:
                SEGMENT_CACHE[m] = t[["ID", "Segment"]].copy()
            desired = ["ID", "기준년월"] + FEATS
            keep = [c for c in desired if c in t.columns]
            dfs.append(t[keep])
        merged = dfs[0]
        for tdf in dfs[1:]:
            merged = merged.merge(tdf, on=["ID", "기준년월"], how="left")
        monthly[m] = merged
    return pd.concat(monthly.values(), ignore_index=True)

def preprocess_train(df):
    vid = df.groupby("ID")["기준년월"].nunique().eq(len(MONTHS))
    df = df[df["ID"].isin(vid[vid].index)]
    lab = SEGMENT_CACHE[MONTHS[-1]][SEGMENT_CACHE[MONTHS[-1]]["ID"].isin(df["ID"])]
    df = df[df["ID"].isin(lab["ID"])].sort_values(["ID", "기준년월"])
    y = torch.tensor(lab["Segment"].astype("category").cat.codes.values, dtype=torch.long)
    scaler_dict = {}
    for col in FEATS:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes
            df[col] = pd.to_numeric(df[col], downcast='float')
            if df[col].isnull().all():
                df[col] = 0
                sc = StandardScaler()
                sc.mean_ = np.array([0.]); sc.scale_ = np.array([1.])
            else:
                df[col] = df[col].fillna(0)
                sc = StandardScaler()
                df[col] = df.groupby("기준년월")[col].transform(lambda s: np.zeros(len(s)) if s.std() == 0 else sc.fit_transform(s.values.reshape(-1, 1)).flatten())
            scaler_dict[col] = sc
    joblib.dump(scaler_dict, "scaler.pkl")
    seq = df.groupby("ID")[FEATS].apply(lambda x: x.to_numpy()).values
    X = torch.tensor(np.stack(seq), dtype=torch.float32)
    return X, y

def preprocess_test(df):
    scaler_dict = joblib.load("scaler.pkl")
    for col in FEATS:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category').cat.codes
            df[col] = pd.to_numeric(df[col], downcast='float')
            df[col] = df[col].fillna(0)
    for col in FEATS:
        if col in df.columns:
            sc = scaler_dict[col]
            if not hasattr(sc, 'mean_'):
                sc.mean_ = np.array([0.])
                sc.scale_ = np.array([1.])
            df[col] = df.groupby("기준년월")[col].transform(lambda s: sc.transform(s.values.reshape(-1, 1)).flatten())
    grouped = df.groupby("ID")[FEATS]
    seq = grouped.apply(lambda x: x.to_numpy()).values
    ids = grouped.apply(lambda x: x.name).tolist()
    X = torch.tensor(np.stack(seq), dtype=torch.float32)
    return X, ids

class TransformerClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, emb_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.input_proj(x)
        h = self.transformer(x)
        h = h.mean(dim=1)
        return self.cls_head(h)

if __name__ == '__main__':
    device = get_device()
    torch.backends.cudnn.benchmark = True

    print(" 병합 및 전처리 (train)...")
    df_train = load_monthly(BASE_TRAIN, mode="train")
    X, y = preprocess_train(df_train)
    X, y = X.to(device), y.to(device)
    print_diagnostics(X, y, verbose=False)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kfold.split(X.cpu(), y.cpu())):
        print(f"\n▶ Fold {fold+1}")
        X_train_os, y_train_os = oversample(X[tr_idx], y[tr_idx])
        tr_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_os, y_train_os),
            batch_size=2048,
            shuffle=True
            )

        va_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[val_idx], y[val_idx]), batch_size=2048, shuffle=False)

        model = TransformerClassifier(feat_dim=X.shape[2], num_classes=len(y.unique())).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60 * len(tr_loader))

        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        patience, patience_limit = 0, 20
        for ep in range(1, 61):
            model.train(); loss_sum = 0
            for xb, yb in tr_loader:
                opt.zero_grad()
                with autocast(device.type):
                    out = model(xb)
                    loss = crit(out, yb)
                if torch.isnan(loss):
                    break
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                loss_sum += loss.item() * len(yb)
            train_losses.append(loss_sum / len(tr_idx))

            model.eval(); val_loss_sum = 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    with autocast(device.type):
                        out = model(xb); val_loss = crit(out, yb)
                    if torch.isnan(val_loss):
                        break
                    val_loss_sum += val_loss.item() * len(yb)
            val_loss_avg = val_loss_sum / len(val_idx)
            val_losses.append(val_loss_avg)

            print(f"Epoch{ep:03d}|TrainLoss{train_losses[-1]:.4f}|ValLoss{val_loss_avg:.4f}")

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                torch.save(model.state_dict(), f"model_best_fold{fold+1}.pt")
                patience = 0
            elif ep > 10:
                patience += 1
                if patience >= patience_limit:
                    print(" Early stopping triggered")
                    break
        print(f"▶ Fold {fold+1} 완료 | Best Val Loss: {best_val_loss:.4f}")

    print("▶ 병합 및 전처리 (test)...")
    df_test = load_monthly(BASE_TEST, mode="test")
    X_test, ids_test = preprocess_test(df_test)
    X_test = X_test.to(device)

    model = TransformerClassifier(feat_dim=X_test.shape[2], num_classes=len(y.unique())).to(device)
    model.load_state_dict(torch.load("model_best_fold1.pt"))  # 예시로 첫 번째 폴드 사용
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 1024):
            xb = X_test[i:i+1024]
            out = model(xb)
            preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds).numpy()

    # 예측 결과 저장 (Segment를 a, b, c, d, e로 매핑)
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    mapped_preds = [label_map[int(p)] for p in preds]
    result_df = pd.DataFrame({"ID": ids_test, "Segment": mapped_preds})
    try:
        result_df.to_csv("submission.csv", index=False, encoding="utf-8")
    except PermissionError:
        print(" 'submission.csv' 파일이 열려 있어 덮어쓸 수 없습니다. 다른 이름으로 저장합니다.")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fallback_path = f"submission_{timestamp}.csv"
        result_df.to_csv(fallback_path, index=False, encoding="utf-8")
        print(f" '{fallback_path}'로 저장 완료")
    else:
        print(" 테스트 예측 완료: submission.csv 저장됨")

    print(f" 예측 클래스 분포: {np.unique(mapped_preds, return_counts=True)}")
    print(f" 총 {len(mapped_preds)}건의 예측을 완료했습니다.")

    # 결과 시각화 (1 fold 기준)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve (Fold 1)')
    plt.legend(); plt.tight_layout(); plt.show()