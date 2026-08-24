"""
复用已有的 day_dict + embedding，只重新生成 CAPM 5 日异质波动率（IVOL）标签并重建样本。

作用:
    - 不用重跑 GLiNER/FinBERT，直接基于 data_processing/day_dict.pt 的文本张量
      生成新的 train/val/test_samples.pt。
    - 标签：sector 等权日收益相对 S&P 500 的 CAPM 滚动 beta 残差，未来 5 日 std，
      按全局中位数或截面分位点二分类（高波动 = 1）。

用法:
    # 全局中位数（默认）
    python data_processing/scripts/build_capm_ivol5_samples.py \
        --stocks stocks.csv \
        --sp500 data_processing/sp500.csv \
        --day_dict data_processing/day_dict.pt \
        --src_config data_processing/config.pt \
        --out_dir data_processing/samples_capm_ivol5 \
        --val_start_date 2023-01-01 \
        --test_start_date 2023-07-01

    # 截面 top 20%（高于当日 80% 分位点）
    python data_processing/scripts/build_capm_ivol5_samples.py \
        --stocks stocks.csv \
        --sp500 data_processing/sp500.csv \
        --day_dict data_processing/day_dict.pt \
        --src_config data_processing/config.pt \
        --xs_median --xs_quantile 0.8 \
        --val_start_date 2023-01-01 \
        --test_start_date 2023-07-01

输出:
    data_processing/samples_capm_ivol5/{train,val,test}_samples.pt + config.pt
    若使用 --xs_median，输出目录会追加 _top{N} 后缀，如 samples_capm_ivol5_top20。
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

from helpers import (
    compute_idiosyncratic_vol, build_time_series_samples,
    temporal_train_val_test_split,
)


def build_sector_returns(df_stocks):
    df = df_stocks.copy()
    df['RET'] = pd.to_numeric(df['RET'], errors='coerce')
    return (
        df.groupby(['sector', 'date'])['RET']
          .mean()
          .reset_index()
          .rename(columns={'RET': 'sector_RET'})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stocks', default='stocks.csv')
    parser.add_argument('--sp500', default='data_processing/sp500.csv')
    parser.add_argument('--day_dict', default='data_processing/day_dict.pt')
    parser.add_argument('--src_config', default='data_processing/config.pt')
    parser.add_argument('--out_dir', default='data_processing/samples_capm_ivol5')
    parser.add_argument('--W', type=int, default=20)
    parser.add_argument('--beta_window', type=int, default=60,
                        help='CAPM beta 滚动估计窗口（交易日）')
    parser.add_argument('--val_start_date', default='2023-01-01')
    parser.add_argument('--test_start_date', default='2023-07-01')
    parser.add_argument('--xs_median', action='store_true',
                        help='使用截面分位点标签（每天独立），否则用全局中位数')
    parser.add_argument('--xs_quantile', type=float, default=0.5,
                        help='截面标签的 threshold 分位点（top 20% = 0.8；top 50% = 0.5）')
    args = parser.parse_args()

    out_dir = args.out_dir
    if args.xs_median:
        # 根据分位点命名输出目录，例如 top20 / top50
        q_label = f"top{round((1 - args.xs_quantile) * 100)}"
        out_dir = out_dir.rstrip('/') + f'_{q_label}'
    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载 day_dict
    print(f"加载 day_dict: {args.day_dict}")
    day_dict = torch.load(args.day_dict, weights_only=False)
    print(f"  {len(day_dict)} 个 (sector, date) 条目, {len({k[0] for k in day_dict})} 个 sector")

    # 2. 计算 CAPM IVOL 标签
    df_stocks = pd.read_csv(args.stocks)
    df_stocks['date'] = pd.to_datetime(df_stocks['date']).dt.date
    df_sp500 = pd.read_csv(args.sp500)
    df_sp500['date'] = pd.to_datetime(df_sp500['date']).dt.date

    sector_ret = build_sector_returns(df_stocks)
    sector_label = compute_idiosyncratic_vol(
        sector_ret,
        df_sp500,
        window=5,
        beta_window=args.beta_window,
        ret_col='sector_RET',
        mkt_col='SP500_RET',
        ticker_col='sector',
        date_col='date',
    )
    sector_label = sector_label.dropna(subset=['ivol_5'])
    print(f"IVOL 标签: {len(sector_label)} 行, 全局中位数 {sector_label['ivol_5'].median():.6f}")

    # 3. 与 day_dict 文本覆盖范围对齐：取 day_dict 日期范围，并留出 W 天 lookback
    label_df = sector_label[['sector', 'date', 'ivol_5']].copy()
    label_df['date'] = pd.to_datetime(label_df['date'])

    dd_sectors = sorted({k[0] for k in day_dict.keys()})
    dd_dates = sorted({k[1] for k in day_dict.keys()})
    text_start = min(dd_dates) + pd.tseries.offsets.BDay(args.W)
    text_end = max(dd_dates)
    print(f"文本覆盖: {min(dd_dates).date()} ~ {text_end.date()}, 参与训练日期 {text_start.date()} ~ {text_end.date()}")
    label_df = label_df[(label_df['date'] >= text_start) & (label_df['date'] <= text_end)]

    # 只保留 day_dict 里实际有的 sector，避免截面中位数基准偏移
    before = label_df['sector'].nunique()
    label_df = label_df[label_df['sector'].isin(dd_sectors)]
    print(f"sector 对齐: {before} -> {label_df['sector'].nunique()} 个")

    # 标签：时间序列（全局中位数）或截面分位点
    if args.xs_median:
        q = args.xs_quantile
        label_df['xs_ivol'] = label_df['ivol_5'] - label_df.groupby('date')['ivol_5'].transform(lambda x: x.quantile(q))
        label_col = 'xs_ivol'
        threshold = 0.0  # margin 已经以 0 为界
        print(f"已生成截面分位点标签: top {round((1 - q) * 100)}% (quantile={q:.2f})")
    else:
        label_col = 'ivol_5'
        threshold = None  # 全局中位数
        print(f"已生成全局中位数标签: {label_col}")

    # 4. 构建样本
    samples = build_time_series_samples(
        label_df,
        day_dict,
        W=args.W,
        label_col=label_col,
        ticker_col='sector',
        date_col='date',
        threshold=threshold,
    )
    train_s, val_s, test_s = temporal_train_val_test_split(
        samples, args.val_start_date, args.test_start_date
    )

    # 5. 保存
    shutil.copy(args.src_config, os.path.join(out_dir, 'config.pt'))
    for name, split in [('train', train_s), ('val', val_s), ('test', test_s)]:
        torch.save(split, os.path.join(out_dir, f'{name}_samples.pt'))
        dist = Counter(s[3] for s in split)
        n = max(len(split), 1)
        dates = sorted(s[1] for s in split)
        rng = f"{dates[0].date()} ~ {dates[-1].date()}" if dates else "-"
        print(f"  {name:<5} {len(split):>5} 样本 | 高波动(1) {dist[1]/n:.1%} | {rng}")

    print(f"\n✅ 已写出 {out_dir}/")


if __name__ == '__main__':
    main()
