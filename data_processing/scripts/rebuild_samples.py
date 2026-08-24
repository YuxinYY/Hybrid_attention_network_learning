"""
在不重跑 GLiNER / FinBERT 的前提下，用新的标签定义重建训练样本。

⚠️ 历史流程（FF3 margin 标签）:
    当前主流程已改为 pipeline.py 内置的 CAPM 中位数二分类标签（capm），
    train/val/test_samples.pt 直接由 pipeline.py 输出到 data_dir。
    本脚本仅保留给旧的 FF3 标签（ts_rise / xs_median / xs_demeaned）作为历史参考，
    需要搭配本地旧的 sector_ivol_labels.csv（FF3 版）使用。

用法:
    python data_processing/scripts/rebuild_samples.py --label_def ts_rise
    python data_processing/scripts/rebuild_samples.py --label_def xs_median

输出:
    data_processing/samples_<label_def>/{train,val,test}_samples.pt + config.pt
"""

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as project_config
from helpers import build_time_series_samples, temporal_train_val_test_split

args_cfg = project_config.args


def main():
    p = argparse.ArgumentParser(description="用新标签重建样本（复用已有 day_dict）")
    p.add_argument("--label_def", required=True,
                   choices=["ts_rise", "xs_median", "xs_demeaned"],
                   help="ts_rise: 高于该行业当前水平; xs_median: 高于同日其他行业中位数; "
                        "xs_demeaned: 相对自身水平的变化幅度高于同日其他行业中位数")
    p.add_argument("--labels", default="data_processing/sector_ivol_labels.csv")
    p.add_argument("--day_dict", default=os.path.join(args_cfg.data_dir, "day_dict.pt"))
    p.add_argument("--src_config", default=os.path.join(args_cfg.data_dir, "config.pt"))
    p.add_argument("--out_dir", default=None, help="默认 <data_dir>/samples_<label_def>")
    p.add_argument("--W", type=int, default=20, help="lookback 天数")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args_cfg.data_dir, f"samples_{args.label_def}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"加载 day_dict: {args.day_dict} ...")
    day_dict = torch.load(args.day_dict, weights_only=False)
    dd_sectors = sorted({k[0] for k in day_dict.keys()})
    print(f"day_dict: {len(day_dict)} 个 (sector, date) 条目, {len(dd_sectors)} 个 sector")

    labels = pd.read_csv(args.labels)
    labels["date"] = pd.to_datetime(labels["date"])
    before = labels["sector"].nunique()
    # 只保留有文本 embedding 的 sector，避免造出永远全零输入的样本
    labels = labels[labels["sector"].isin(dd_sectors)]
    print(f"标签表: {len(labels)} 行, sector {before} -> {labels['sector'].nunique()} "
          f"(与 day_dict 对齐), {labels['date'].min().date()} ~ {labels['date'].max().date()}")

    # 标签能算到的日期比文本覆盖范围更早（估 beta 用了 2020 年数据），但那些 anchor 的
    # lookback 窗口落在 day_dict 之外，输入会是全零。这里按文本覆盖范围裁掉，
    # 并留出 W 个交易日的 lookback 余量。
    dd_dates = pd.to_datetime(sorted({k[1] for k in day_dict.keys()}))
    text_start = dd_dates.min() + pd.tseries.offsets.BDay(args.W)
    text_end = dd_dates.max()
    n_before = len(labels)
    labels = labels[(labels["date"] >= text_start) & (labels["date"] <= text_end)]
    print(f"按 day_dict 文本覆盖裁剪 ({dd_dates.min().date()} ~ {text_end.date()}, "
          f"留 {args.W} 日 lookback): {n_before} -> {len(labels)} 行")

    # 截面中位数必须在"实际参与训练的 sector"上重算：build_ivol_labels.py 是在全部 11 个
    # sector 上算的，而 day_dict 只覆盖 10 个，直接用会让基准偏移、正负类不再各占一半
    if args.label_def == "xs_median":
        labels["xs_median"] = (labels["ivol_fwd"]
                               - labels.groupby("date")["ivol_fwd"].transform("median"))
        print(f"已在 {labels['sector'].nunique()} 个 sector 上重算截面中位数基准")
    elif args.label_def == "xs_demeaned":
        labels["xs_demeaned"] = (labels["log_ratio"]
                                 - labels.groupby("date")["log_ratio"].transform("median"))
        print(f"已在 {labels['sector'].nunique()} 个 sector 上重算截面中位数基准")

    label_df = labels[["sector", "date", args.label_def]].dropna()

    samples = build_time_series_samples(
        label_df,
        day_dict,
        W=args.W,
        label_col=args.label_def,
        ticker_col="sector",
        date_col="date",
        threshold=0.0,  # margin 已经以 0 为分界
    )
    train_s, val_s, test_s = temporal_train_val_test_split(
        samples, args_cfg.val_start_date, args_cfg.test_start_date
    )

    for name, split in [("train", train_s), ("val", val_s), ("test", test_s)]:
        torch.save(split, os.path.join(out_dir, f"{name}_samples.pt"))
        dist = Counter(s[3] for s in split)
        n = max(len(split), 1)
        dates = sorted(s[1] for s in split)
        rng = f"{dates[0].date()} ~ {dates[-1].date()}" if dates else "-"
        print(f"  {name:<6} {len(split):>5} 样本 | 正类 {dist[1]}/{len(split)} "
              f"({dist[1]/n:.1%}) | {rng}")

    shutil.copy(args.src_config, os.path.join(out_dir, "config.pt"))
    print(f"\n✅ 已写出 {out_dir}/ (train/val/test_samples.pt + config.pt)")


if __name__ == "__main__":
    main()
