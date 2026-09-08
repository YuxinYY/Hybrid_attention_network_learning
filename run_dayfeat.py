"""
用日级统计量特征跑简化模型，并与 Gradient Boosting 对照。

与 run.py 的区别:
    run.py   输入 (B, W, 50, 768) 文本 embedding -> HAN（含 post-level attention），44 万参数
    本脚本   输入 (B, W, F) 日级统计量           -> DayStatGRU（无 post-level attention），约 5 万参数
             另外并列跑一个 HistGradientBoosting 作为非神经网络对照

用法:
    python run_dayfeat.py --label_def capm
    python run_dayfeat.py --label_def all

注意:
    - 样本由 pipeline.py 直接输出到 data_dir（config.py 指定），
      不再使用 rebuild_samples 的 samples_<label_def> 目录。
    - 日级统计量特征由 build_day_features.py 生成，同样位于 data_dir。
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from model import DayStatGRU
import config as project_config

DAY_FEATURES = os.path.join(project_config.args.data_dir, "day_features.parquet")
SAMPLE_DIR = project_config.args.data_dir  # pipeline.py 输出的 train/val/test_samples.pt
LABEL_DEFS = ["capm"]


def load_sequences(label_def, W=20):
    """把每个样本的 20 天 lookback 展开成 (W, F) 的统计量序列。"""
    day = pd.read_parquet(DAY_FEATURES)
    day["date"] = pd.to_datetime(day["date"])
    feat_cols = [c for c in day.columns if c not in ("sector", "date")]
    lut = {(r.sector, r.date): i for i, r in enumerate(day.itertuples())}
    mat = day[feat_cols].to_numpy(dtype=np.float32)
    zero = np.zeros(len(feat_cols), dtype=np.float32)

    out = {}
    for split in ["train", "val", "test"]:
        samples = torch.load(os.path.join(SAMPLE_DIR, f"{split}_samples.pt"),
                             weights_only=False)
        X = np.stack([
            np.stack([mat[lut[(sec, pd.Timestamp(d))]] if (sec, pd.Timestamp(d)) in lut else zero
                      for d in lb])
            for sec, _a, lb, _y in samples
        ])
        y = np.array([s[3] for s in samples], dtype=np.int64)
        cover = np.mean([[(sec, pd.Timestamp(d)) in lut for d in lb]
                         for sec, _a, lb, _y in samples])
        out[split] = (X, y)
        print(f"  {split}: X={X.shape}  正类={y.mean():.1%}  lookback 日覆盖率={cover:.1%}")
    return out, feat_cols


def standardize(data):
    """用 train 的均值方差归一化，避免验证/测试信息泄漏。"""
    Xtr = data["train"][0]
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0)
    sd[sd < 1e-6] = 1.0
    return {k: ((X - mu) / sd, y) for k, (X, y) in data.items()}


def report(name, y, pred, prob):
    auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
    print(f"  {name:<28} acc={accuracy_score(y, pred):.4f}  "
          f"macro_f1={f1_score(y, pred, average='macro'):.4f}  auc={auc:.4f}")
    return {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average='macro'),
            "auc": auc}


def run_gbm(data):
    """非神经网络对照：把序列压成紧凑特征后跑 HistGradientBoosting。"""
    def flat(X):
        # 原始 20 天全展开会有 420 维，对 4,640 个样本太多；改用有含义的时间摘要
        return np.concatenate([
            X.mean(1), X.std(1), X[:, -5:].mean(1), X[:, -5:].mean(1) - X[:, :-5].mean(1),
            X[:, -1], X.max(1) - X.min(1),
        ], axis=1)

    Xtr, ytr = flat(data["train"][0]), data["train"][1]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=15, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.15,
                                         random_state=42, class_weight="balanced")
    clf.fit(Xtr, ytr)
    res = {}
    for split in ["val", "test"]:
        X, y = flat(data[split][0]), data[split][1]
        res[split] = report(f"GBM ({split})", y, clf.predict(X), clf.predict_proba(X)[:, 1])
    return res


def run_nn(data, feature_dim, epochs=40, patience=8, seed=0):
    torch.manual_seed(seed)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def loader(split, train=False):
        X, y = data[split]
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        if train:
            cnt = np.maximum(np.bincount(y, minlength=2), 1)
            w = torch.tensor((1.0 / cnt)[y], dtype=torch.float)
            return DataLoader(ds, batch_size=64, drop_last=True,
                              sampler=WeightedRandomSampler(w, len(w), replacement=True))
        return DataLoader(ds, batch_size=256, shuffle=False)

    tr, va, te = loader("train", True), loader("val"), loader("test")
    model = DayStatGRU(feature_dim=feature_dim).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  DayStatGRU 参数量: {n_par:,}  (参数/样本 = {n_par/len(data['train'][1]):.1f} : 1)")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    crit = torch.nn.CrossEntropyLoss()

    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb)[0], yb)
            loss.backward()
            opt.step()

        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for xb, yb in va:
                probs.append(torch.softmax(model(xb.to(device))[0], -1)[:, 1].cpu().numpy())
                ys.append(yb.numpy())
        auc = roc_auc_score(np.concatenate(ys), np.concatenate(probs))
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"  早停于 epoch {ep+1} (最佳 val AUC={best_auc:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    res = {}
    for split, dl in [("val", va), ("test", te)]:
        probs, ys = [], []
        with torch.no_grad():
            for xb, yb in dl:
                probs.append(torch.softmax(model(xb.to(device))[0], -1)[:, 1].cpu().numpy())
                ys.append(yb.numpy())
        p, y = np.concatenate(probs), np.concatenate(ys)
        res[split] = report(f"DayStatGRU ({split})", y, (p > 0.5).astype(int), p)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_def", default="capm", choices=["capm", "all"])
    args = ap.parse_args()
    defs = LABEL_DEFS if args.label_def == "all" else [args.label_def]

    summary = {}
    for ld in defs:
        print(f"\n{'='*64}\n📊 {ld}\n{'='*64}")
        data, feat_cols = load_sequences(ld)
        data = standardize(data)
        print(f"  特征维度 F={len(feat_cols)}")
        print("\n  --- Gradient Boosting (非神经网络对照) ---")
        gbm = run_gbm(data)
        print("\n  --- DayStatGRU (简化神经网络) ---")
        nn_res = run_nn(data, feature_dim=len(feat_cols))
        summary[ld] = {"GBM": gbm["test"], "DayStatGRU": nn_res["test"]}

    print(f"\n{'='*64}\n📋 测试集汇总\n{'='*64}")
    print(f"{'标签':<14}{'模型':<14}{'acc':>8}{'macro_f1':>10}{'auc':>8}")
    for ld, models in summary.items():
        for name, m in models.items():
            print(f"{ld:<14}{name:<14}{m['acc']:>8.4f}{m['macro_f1']:>10.4f}{m['auc']:>8.4f}")


if __name__ == "__main__":
    main()
