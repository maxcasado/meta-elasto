
from pathlib import Path
import argparse
import re
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


FNAME_RE = re.compile(r"Elastome_(\d+)_([A-Za-z]+)_angle_(\d+)\.csv", re.IGNORECASE)


def read_metadata(xlsx_path: Path) -> Dict[int, int]:
    df = pd.read_excel(xlsx_path)
    a = df.iloc[:, 0].astype(int).to_numpy()
    b = df.iloc[:, 1].astype(int).to_numpy()
    return {int(pid): int(score) for pid, score in zip(a, b)}


def read_csv_timeseries(p: Path) -> np.ndarray:
    df = pd.read_csv(p)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"No numeric columns in {p}")
    return num.to_numpy().T


def discover_samples(root: Path) -> List[Dict]:
    res_dir = root / "Skin" / "Resultados"
    xlsx = (root / "Skin" / "0list.xlsx")
    if not res_dir.exists():
        raise FileNotFoundError(f"Resultados folder not found: {res_dir}")
    if not xlsx.exists():
        raise FileNotFoundError(f"0list.xlsx not found: {xlsx}")
    labels = read_metadata(xlsx)
    items = []
    for file in sorted(res_dir.glob("Elastome_*_*.csv")):
        m = FNAME_RE.match(file.name)
        if not m:
            continue
        pid = int(m.group(1))
        kind = m.group(2).lower()
        angle = int(m.group(3))
        arr = read_csv_timeseries(file)
        if kind.startswith("healthy"):
            y = 0
        else:
            if pid not in labels:
                raise KeyError(f"Patient {pid} has no label in 0list.xlsx")
            y = int(labels[pid])
            if y == 0:
                y = 1
        items.append({"path": str(file), "pid": pid, "angle": angle, "X": arr, "y": y})
    return items


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return (x - m) / s


def pad_or_trim(x: np.ndarray, L: int) -> np.ndarray:
    C, T = x.shape
    if T == L:
        return x
    if T > L:
        return x[:, :L]
    out = np.zeros((C, L), dtype=x.dtype)
    out[:, :T] = x
    return out


def basic_features(x: np.ndarray, n_fft_bands: int = 6) -> np.ndarray:
    x = zscore_per_channel(x)
    C, T = x.shape
    feats = []
    feats.append(x.mean(axis=1))
    feats.append(x.std(axis=1))
    feats.append(np.median(x, axis=1))
    feats.append(np.percentile(x, 25, axis=1))
    feats.append(np.percentile(x, 75, axis=1))
    feats.append(np.max(x, axis=1) - np.min(x, axis=1))
    Xf = np.fft.rfft(x, axis=1)
    psd = (Xf * np.conj(Xf)).real
    psd_sum = psd.sum(axis=1, keepdims=True) + 1e-12
    psd_norm = psd / psd_sum
    Lf = psd.shape[1]
    edges = np.linspace(0, Lf, n_fft_bands + 1, dtype=int)
    band_energies = []
    for i in range(n_fft_bands):
        band = psd_norm[:, edges[i]:edges[i + 1]].sum(axis=1)
        band_energies.append(band)
    feats.extend(band_energies)
    f_idx = np.arange(Lf)[None, :]
    centroid = (psd_norm * f_idx).sum(axis=1)
    feats.append(centroid)
    return np.concatenate(feats, axis=0)


def build_tabular_dataset(items: List[Dict], n_fft_bands: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, pid_list = [], [], []
    for it in items:
        f = basic_features(it["X"], n_fft_bands=n_fft_bands)
        X_list.append(f)
        y_list.append(it["y"])
        pid_list.append(it["pid"])
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    pids = np.array(pid_list, dtype=int)
    return X, y, pids


def loso_splits(pids: np.ndarray):
    unique = np.unique(pids)
    for u in unique:
        test_idx = np.where(pids == u)[0]
        train_idx = np.where(pids != u)[0]
        yield train_idx, test_idx


def train_eval_tabular(X: np.ndarray, y: np.ndarray, pids: np.ndarray, model_name: str) -> Dict:
    if model_name == "lgbm" and HAS_LGBM:
        clf = lgb.LGBMClassifier(objective="multiclass", num_leaves=31, learning_rate=0.05, n_estimators=300)
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif model_name == "rf":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=400, random_state=42))])
    elif model_name == "svm":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=5, gamma="scale"))])
    elif model_name == "logreg":
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial"))])
    else:
        if HAS_LGBM:
            clf = lgb.LGBMClassifier(objective="multiclass", num_leaves=31, learning_rate=0.05, n_estimators=300)
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        else:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=400, random_state=42))])
    y_true_all, y_pred_all = [], []
    folds = []
    for i, (tr, te) in enumerate(loso_splits(pids), start=1):
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[te])
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(yp.tolist())
        folds.append({"fold": i, "support": int(len(te)), "acc": float(accuracy_score(y[te], yp)), "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0))})
    overall = {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
        "per_class_f1": f1_score(y_true_all, y_pred_all, average=None, zero_division=0).tolist(),
        "folds": folds,
    }
    return overall


if HAS_TORCH:
    class WavesDataset(Dataset):
        def __init__(self, items: List[Dict], seq_len: int):
            self.items = items
            self.seq_len = seq_len

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            it = self.items[idx]
            x = pad_or_trim(zscore_per_channel(it["X"]), self.seq_len).astype(np.float32)
            y = int(it["y"])
            return x, y

    class TCNBlock(nn.Module):
        def __init__(self, c_in, c_out, k=7, d=1, p=0.1):
            super().__init__()
            self.pad = nn.ConstantPad1d(((k - 1) * d, 0), 0.0)
            self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d)
            self.norm = nn.BatchNorm1d(c_out)
            self.act = nn.ReLU()
            self.drop = nn.Dropout(p)

        def forward(self, x):
            x = self.pad(x)
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.drop(x)
            return x

    class SimpleTCN(nn.Module):
        def __init__(self, c_in: int, n_classes: int, width: int = 64, layers: int = 4, k: int = 7, p: float = 0.1):
            super().__init__()
            blocks = []
            cin = c_in
            for i in range(layers):
                cout = width
                d = 2 ** i
                blocks.append(TCNBlock(cin, cout, k=k, d=d, p=p))
                cin = cout
            self.net = nn.Sequential(*blocks)
            self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(width, n_classes))

        def forward(self, x):
            return self.head(self.net(x))

    def split_items_loso(items: List[Dict]):
        by_pid = defaultdict(list)
        for it in items:
            by_pid[it["pid"]].append(it)
        sorted_pids = sorted(by_pid.keys())
        for pid in sorted_pids:
            train, test = [], []
            for p2, lst in by_pid.items():
                if p2 == pid:
                    test.extend(lst)
                else:
                    train.extend(lst)
            yield pid, train, test

    def train_eval_tcn(items: List[Dict], num_classes: int, seq_len: int, batch_size: int, epochs: int, lr: float, width: int, layers: int, k: int, dropout: float, device: str) -> Dict:
        device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        y_true_all, y_pred_all = [], []
        folds = []
        for fold_idx, (pid, train_items, test_items) in enumerate(split_items_loso(items), start=1):
            model = SimpleTCN(c_in=train_items[0]["X"].shape[0], n_classes=num_classes, width=width, layers=layers, k=k, p=dropout).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()
            train_ds = WavesDataset(train_items, seq_len=seq_len)
            test_ds = WavesDataset(test_items, seq_len=seq_len)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            for _ in range(epochs):
                model.train()
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = crit(logits, yb)
                    loss.backward()
                    opt.step()
            model.eval()
            yp_fold, yt_fold = [], []
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    preds = logits.argmax(dim=1).cpu().numpy().tolist()
                    yp_fold.extend(preds)
                    yt_fold.extend(yb.numpy().tolist())
            y_true_all.extend(yt_fold)
            y_pred_all.extend(yp_fold)
            folds.append({"fold": fold_idx, "pid": int(pid), "support": len(yt_fold), "acc": float(accuracy_score(yt_fold, yp_fold)), "macro_f1": float(f1_score(yt_fold, yp_fold, average="macro", zero_division=0))})
        overall = {
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
            "per_class_f1": f1_score(y_true_all, y_pred_all, average=None, zero_division=0).tolist(),
            "folds": folds,
        }
        return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--model", type=str, default="lgbm", choices=["lgbm", "rf", "svm", "logreg", "tcn"])
    ap.add_argument("--n_fft_bands", type=int, default=6)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--dump_metrics", type=str, default="")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    items = discover_samples(root)

    if args.model == "tcn":
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        yvals = [it["y"] for it in items]
        n_classes = int(max(yvals) + 1)
        metrics = train_eval_tcn(items, n_classes, args.seq_len, args.batch_size, args.epochs, args.lr, args.width, args.layers, args.kernel, args.dropout, args.device)
    else:
        X, y, pids = build_tabular_dataset(items, n_fft_bands=args.n_fft_bands)
        metrics = train_eval_tabular(X, y, pids, args.model)

    print(json.dumps(metrics, indent=2))
    if args.dump_metrics:
        Path(args.dump_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
