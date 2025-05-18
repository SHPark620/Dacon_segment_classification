import os, time, numpy as np, pandas as pd, torch, joblib
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

BASE = "E:/AI_proj/dacon_segment/open/train"
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
    'Segment', '남녀구분코드', '연령', '회원여부_이용가능', '회원여부_이용가능_CA', '회원여부_이용가능_카드론',
    '소지여부_신용', '소지카드수_유효_신용', '소지카드수_이용가능_신용', '입회경과개월수_신용',
    '동의여부_한도증액안내', '수신거부여부_TM', '수신거부여부_DM', '수신거부여부_메일', '수신거부여부_SMS',
    '탈회횟수_누적', '최종탈회후경과월', '마케팅동의여부', '유효카드수_신용', '유효카드수_체크',
    '이용가능카드수_신용', '이용가능카드수_체크', '이용카드수_신용', '이용카드수_체크',
    '이용금액_R3M_신용', '이용금액_R3M_체크', '_1순위카드이용금액', '_1순위카드이용건수',
    '_1순위신용체크구분', '_2순위카드이용금액', '_2순위카드이용건수', '_2순위신용체크구분',
    '최종카드발급일자', '보유여부_해외겸용_본인', '이용가능여부_해외겸용_본인', '이용여부_3M_해외겸용_본인',
    '보유여부_해외겸용_신용_본인', '이용가능여부_해외겸용_신용_본인', '이용여부_3M_해외겸용_신용_본인',
    '기본연회비_B0M', '제휴연회비_B0M', '할인금액_기본연회비_B0M', '할인금액_제휴연회비_B0M',
    '청구금액_기본연회비_B0M', '청구금액_제휴연회비_B0M', '카드신청건수', '최종카드발급경과월',
    '인입불만횟수_IB_R6M', '홈페이지_금융건수_R3M', '변동률_할부평잔', '변동률_잔액_B1M', '변동률_CA평잔',
    '증감율_이용금액_카드론_분기', '증감율_이용건수_신용_분기', '증감율_이용금액_신용_전월', '평잔_CA_3M',
    '잔액_카드론_B5M', '잔액_현금서비스_B1M', '연체잔액_할부_B0M', '최종연체개월수_R15M',
    'IB문의건수_결제_R6M', 'IB문의건수_결제일변경_R6M', 'IB문의건수_CA_R6M', 'IB문의건수_APP_B0M',
    'IB상담건수_VOC민원_R6M', '방문일수_PC_B0M'
]

SEGMENT_CACHE = {}
scaler = GradScaler()

def load_monthly(base):
    monthly = {}
    for m in MONTHS:
        dfs = []
        for d in DIRS:
            name = FILENAME_MAP[d]
            fp = f"{base}/{d}/{m}_train_{name}.parquet"
            if not os.path.exists(fp):
                raise FileNotFoundError(f" 파일 없음: {fp}")
            t = pd.read_parquet(fp)
            t["기준년월"] = m
            if d == "1.회원정보" and m == "201812" and "Segment" in t.columns:
                SEGMENT_CACHE[m] = t[["ID", "Segment"]].copy()
            desired_cols = ["ID", "기준년월"] + FEATS
            keep_cols = [col for col in desired_cols if col in t.columns]
            dfs.append(t[keep_cols])
        merged = dfs[0]
        for t in dfs[1:]:
            merged = merged.merge(t, on=["ID", "기준년월"], how="left")
        monthly[m] = merged
    return pd.concat(monthly.values(), ignore_index=True)


def preprocess(df, fit_scaler=True, scaler_dict=None):
    vid = df.groupby("ID")["기준년월"].nunique().eq(6)
    valid_ids = vid[vid].index.tolist()
    df = df[df["ID"].isin(valid_ids)]
    if "201812" not in SEGMENT_CACHE:
        raise KeyError("201812월의 'Segment' 정보가 없습니다.")
    lab = SEGMENT_CACHE["201812"][SEGMENT_CACHE["201812"]["ID"].isin(df["ID"])]
    df = df[df["ID"].isin(lab["ID"])].sort_values(["ID", "기준년월"])
    y = torch.tensor(lab["Segment"].astype("category").cat.codes.values, dtype=torch.long)
    df = df[["ID", "기준년월"] + FEATS]
    for col in FEATS:
        if col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype("category").cat.codes
            df[col] = pd.to_numeric(df[col], downcast="float")
    if fit_scaler:
        scaler_dict = {}
        for col in FEATS:
            if col in df.columns:
                sc = StandardScaler()
                def safe_transform(s): return np.zeros_like(s) if s.std()==0 else sc.fit_transform(s.values.reshape(-1,1)).flatten()
                df[col] = df.groupby("기준년월")[col].transform(safe_transform)
                scaler_dict[col] = sc
        joblib.dump(scaler_dict, "scaler.pkl")
    else:
        for col in FEATS:
            if col in df.columns:
                sc = scaler_dict[col]
                df[col] = df.groupby("기준년월")[col].transform(lambda s: sc.transform(s.values.reshape(-1,1)).flatten())
    seq = df.groupby("ID")[FEATS].apply(lambda x: x.to_numpy()).values
    X = torch.tensor(np.stack(seq), dtype=torch.float32)
    return X, y

class MLP_LSTM(nn.Module):
    def __init__(self, in_dim, n_cls, emb=64, lstm=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,emb), nn.ReLU())
        self.lstm = nn.LSTM(emb, lstm, batch_first=True)
        self.cls = nn.Linear(lstm, n_cls)
    def forward(self, x):
        B,T,F = x.size()
        h = self.enc(x.view(B*T, F)).view(B, T, -1)
        _,(hn,_) = self.lstm(h)
        return self.cls(hn.squeeze(0))

def get_device():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if "A5000" in torch.cuda.get_device_name(i): return torch.device(f"cuda:{i}")
        return torch.device("cuda:0")
    return torch.device("cpu")

if __name__ == '__main__':
    device = get_device()
    torch.backends.cudnn.benchmark = True
    print("▶  병합 중...")
    df_train = load_monthly(BASE)
    print("▶  전처리 + 스케일러 fit ...")
    X, y = preprocess(df_train, fit_scaler=True)
    del df_train
    X, y = X.to(device), y.to(device)

    tr_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y.cpu(), random_state=42)
    tr_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[tr_idx], y[tr_idx]), batch_size=2048, shuffle=True)
    va_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X[val_idx], y[val_idx]), batch_size=2048, shuffle=False)

    model = MLP_LSTM(len(FEATS), n_cls=len(y.unique())).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    total_step = 50 * len(tr_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_step)

    best_f1 = 0
    train_losses, val_f1s = [], []
    max_epochs = 300
    for ep in range(1, max_epochs+1):
        model.train(); loss_sum = 0
        for xb, yb in tr_loader:
            opt.zero_grad(set_to_none=True)
            with autocast(device.type):
                out = model(xb)
                loss = crit(out, yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            loss_sum += loss.item() * len(yb)
        avg_loss = loss_sum / len(tr_idx)
        train_losses.append(avg_loss)

        model.eval(); preds, gts = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                with autocast(device.type): out = model(xb)
                preds += out.argmax(1).cpu().tolist(); gts += yb.cpu().tolist()
        f1 = f1_score(gts, preds, average='macro')
        val_f1s.append(f1)
        print(f"Epoch{ep:03d}|loss{avg_loss:.4f}|F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "model_best.pt")

    print(" 학습 완료")
    print(f"Best Macro F1: {best_f1:.4f}")

    # 결과 시각화
    epochs = list(range(1, max_epochs+1))
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Validation F1 over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.show()
