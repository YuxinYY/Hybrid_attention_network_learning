"""
异质波动率预测的对照基线。

为什么必须有:
    单看 HAN 的准确率无法判断 Reddit 文本有没有用。波动率有很强的聚集性和均值回复，
    只用"该行业当前的异质波动水平"这一个数就能拿到不低的准确率。只有 HAN 显著超过
    这条无文本基线，才能说文本提供了增量信息。

两条基线（都用与 HAN 完全相同的 train/val/test 切分和样本集合）:
    1. persistence  只用标签自身的市场侧特征做 Logistic Regression:
         - ivol_past          该行业当前异质波动水平
         - log(ivol_past)
         - ivol_past / 该行业过去 250 日均值   (相对自身的高低)
         - 当日全行业 ivol_past 中位数          (全市场波动 regime)
         - ivol_past 相对同日截面中位数的比值
         - sector one-hot
    2. mean-embedding  把 lookback 窗口内所有帖子的 FinBERT embedding 直接平均成一个
       768 维向量，再做 Logistic Regression。用来检验 HAN 的层级注意力结构是否
       真的比"把文本揉成一个平均向量"更好。

用法:
    python data_processing/scripts/baseline_ivol.py --label_def capm
    python data_processing/scripts/baseline_ivol.py --label_def capm --skip_text
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config


def load_splits(sample_dir):
    out = {}
    for name in ["train", "val", "test"]:
        out[name] = torch.load(os.path.join(sample_dir, f"{name}_samples.pt"),
                               weights_only=False)
    return out


def build_market_features(splits, labels_path):
    """从标签表构造无文本特征。样本元组为 (sector, anchor_date, lookback_dates, label)。"""
    lab = pd.read_csv(labels_path)
    lab["date"] = pd.to_datetime(lab["date"])
    lab = lab.sort_values(["sector", "date"])

    # 相对该行业自身长期水平的高低
    lab["ivol_past_ma250"] = (lab.groupby("sector")["ivol_past"]
                                 .transform(lambda s: s.rolling(250, min_periods=60).mean()))
    lab["rel_own"] = lab["ivol_past"] / lab["ivol_past_ma250"]
    # 全市场波动 regime，以及该行业在当日截面上的相对位置
    lab["xs_med_past"] = lab.groupby("date")["ivol_past"].transform("median")
    lab["rel_xs"] = lab["ivol_past"] / lab["xs_med_past"]

    feat_cols = ["ivol_past", "log_ivol_past", "rel_own", "xs_med_past", "rel_xs"]
    lab["log_ivol_past"] = np.log(lab["ivol_past"].clip(lower=1e-6))

    sectors = sorted(lab["sector"].unique())
    idx = lab.set_index(["sector", "date"])

    data = {}
    for name, samples in splits.items():
        X, y = [], []
        for sec, anchor, _lookback, label in samples:
            key = (sec, pd.Timestamp(anchor))
            if key not in idx.index:
                continue
            row = idx.loc[key]
            vals = [row[c] for c in feat_cols]
            if not np.isfinite(vals).all():
                continue
            onehot = [1.0 if sec == s else 0.0 for s in sectors]
            X.append(list(vals) + onehot)
            y.append(label)
        data[name] = (np.array(X, dtype=float), np.array(y, dtype=int))
        print(f"  {name}: {len(y)} 样本 (原 {len(samples)})")
    return data, feat_cols + [f"sector={s}" for s in sectors]


def build_text_features(splits, day_dict_path):
    """lookback 窗口内所有帖子 embedding 的平均向量。"""
    print(f"加载 day_dict: {day_dict_path} ...")
    day_dict = torch.load(day_dict_path, weights_only=False)

    data = {}
    for name, samples in splits.items():
        X, y = [], []
        for sec, _anchor, lookback, label in samples:
            vecs = []
            for d in lookback:
                entry = day_dict.get((sec, pd.Timestamp(d)))
                if entry is None:
                    continue
                t = entry["text"] if isinstance(entry, dict) else entry
                if t is None or len(t) == 0:
                    continue
                vecs.append(np.asarray(t, dtype=np.float32).mean(axis=0))
            # 整个 lookback 窗口一条帖子都没有 -> 用零向量，与 HAN 的 mask 行为一致
            X.append(np.mean(vecs, axis=0) if vecs else np.zeros(768, dtype=np.float32))
            y.append(label)
        data[name] = (np.array(X, dtype=float), np.array(y, dtype=int))
        print(f"  {name}: {len(y)} 样本, 有文本的比例 "
              f"{(np.abs(data[name][0]).sum(axis=1) > 0).mean():.1%}")
    del day_dict
    return data


def evaluate(data, tag, C=1.0):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data["train"], data["val"], data["test"]
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=C, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)

    print(f"\n{'='*56}\n📊 {tag}\n{'='*56}")
    for name, (X, y) in [("val", (Xva, yva)), ("test", (Xte, yte))]:
        Xs = scaler.transform(X)
        pred = clf.predict(Xs)
        prob = clf.predict_proba(Xs)[:, 1]
        auc = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan")
        print(f"  {name:<5} acc={accuracy_score(y, pred):.4f}  "
              f"macro_f1={f1_score(y, pred, average='macro'):.4f}  "
              f"pos_f1={f1_score(y, pred, pos_label=1, zero_division=0):.4f}  "
              f"auc={auc:.4f}  正类占比={y.mean():.1%}")
    print("\n  test confusion matrix (行=真实, 列=预测):")
    print("  " + str(confusion_matrix(yte, clf.predict(scaler.transform(Xte)))).replace("\n", "\n  "))
    print("\n" + classification_report(yte, clf.predict(scaler.transform(Xte)),
                                       target_names=["LowIVOL (0)", "HighIVOL (1)"],
                                       digits=4))
    return clf, scaler


def main():
    p = argparse.ArgumentParser(description="IVOL 预测的无文本 / 平均文本基线")
    p.add_argument("--label_def", required=True, choices=["capm"],
                   help="capm = pipeline.py 的中位数二分类标签（当前唯一标签定义）")
    p.add_argument("--labels", default=os.path.join(
        project_config.args.data_dir, "sector_ivol_labels_capm_2020_2026.csv"))
    p.add_argument("--day_dict", default=os.path.join(
        project_config.args.data_dir, "day_dict.pt"))
    p.add_argument("--sample_dir", default=None)
    p.add_argument("--skip_text", action="store_true", help="跳过 mean-embedding 基线（省内存）")
    args = p.parse_args()

    # 样本由 pipeline.py 直接输出到 data_dir；如无特殊指定就复用同一目录
    sample_dir = args.sample_dir or project_config.args.data_dir
    print(f"标签定义: {args.label_def}, 样本目录: {sample_dir}")
    splits = load_splits(sample_dir)

    print("\n构造市场侧特征 ...")
    mkt, names = build_market_features(splits, args.labels)
    evaluate(mkt, f"Baseline 1 / persistence（无文本, {len(names)} 维）— {args.label_def}")

    if not args.skip_text:
        print("\n构造 mean-embedding 特征 ...")
        txt = build_text_features(splits, args.day_dict)
        evaluate(txt, f"Baseline 2 / mean-embedding（768 维平均向量）— {args.label_def}")


if __name__ == "__main__":
    main()
