"""
复用已有的 day_dict + embedding，只重新生成 CAPM 5 日异质波动率（IVOL）标签并重建样本。

作用:
    - 不用重跑 GLiNER/FinBERT，直接基于 data_processing/day_dict.pt 的文本张量
      生成新的 train/val/test_samples.pt。
    - 标签：sector 等权日收益相对 S&P 500 的 CAPM 滚动 beta 残差，未来 5 日 std，
      按全局中位数二分类（高于中位数 = 高波动 1，否则 0）。

用法:
    python data_processing/scripts/build_capm_ivol5_samples.py \
        --stocks stocks.csv \
        --sp500 data_processing/sp500.csv \
        --day_dict data_processing/day_dict.pt \
        --src_config data_processing/config.pt \
        --out_dir data_processing/samples_capm_ivol5 \
        --val_start_date 2023-01-01 \
        --test_start_date 2023-07-01

输出:
    data_processing/samples_capm_ivol5/{train,val,test}_samples.pt + config.pt
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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

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

    dd_dates = sorted({k[1] for k in day_dict.keys()})
    text_start = min(dd_dates) + pd.tseries.offsets.BDay(args.W)
    text_end = max(dd_dates)
    print(f"文本覆盖: {min(dd_dates).date()} ~ {text_end.date()}, 参与训练日期 {text_start.date()} ~ {text_end.date()}")
    label_df = label_df[(label_df['date'] >= text_start) & (label_df['date'] <= text_end)]

    # 4. 构建样本
    samples = build_time_series_samples(
        label_df,
        day_dict,
        W=args.W,
        label_col='ivol_5',
        ticker_col='sector',
        date_col='date',
        threshold=None,  # 全局中位数二分类
    )
    train_s, val_s, test_s = temporal_train_val_test_split(
        samples, args.val_start_date, args.test_start_date
    )

    # 5. 保存
    shutil.copy(args.src_config, os.path.join(args.out_dir, 'config.pt'))
    for name, split in [('train', train_s), ('val', val_s), ('test', test_s)]:
        torch.save(split, os.path.join(args.out_dir, f'{name}_samples.pt'))
        dist = Counter(s[3] for s in split)
        n = max(len(split), 1)
        dates = sorted(s[1] for s in split)
        rng = f"{dates[0].date()} ~ {dates[-1].date()}" if dates else "-"
        print(f"  {name:<5} {len(split):>5} 样本 | 高波动(1) {dist[1]/n:.1%} | {rng}")

    print(f"\n✅ 已写出 {args.out_dir}/")


if __name__ == '__main__':
    main()
