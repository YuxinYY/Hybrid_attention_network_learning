"""
用未被截断的真实帖子量替换 day_dict 里的 day_features。

为什么需要:
    pipeline.py 在构建 day_dict 之前先按 (sector, date) 做了 DAY_CAP=150 的限额，
    而 build_day_dict_compact 记录的 raw_count 是**限额之后**的数量。结果 72% 的
    sector-day 帖子量都被压成 150，log1p(count) 几乎变成常数（中位数/75分位/95分位全是 150），
    帖子量这个特征等于被废掉了。

    真实数量仍保留在限额前的匹配缓存 temp_matched_features.parquet 里，
    这里把它取回来重新写进 day_dict，不需要重跑任何模型推理。

新的 day_features (D=2):
    [ log1p(当日该行业真实帖子数), log1p(当日 score 总和) ]

用法:
    python data_processing/scripts/patch_day_features.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config as project_config

DATA = project_config.args.data_dir


def main():
    p = argparse.ArgumentParser(description="用真实帖子量修正 day_dict 的 day_features")
    p.add_argument("--day_dict", default=os.path.join(DATA, "day_dict.pt"))
    p.add_argument("--features", default=os.path.join(DATA, "temp_matched_features.parquet"))
    p.add_argument("--output", default=os.path.join(DATA, "day_dict_volfix.pt"))
    args = p.parse_args()

    print(f"读取限额前匹配缓存: {args.features}")
    feat = pd.read_parquet(args.features, columns=["id", "date", "sector", "score"])
    feat["date"] = pd.to_datetime(feat["date"])
    # 一条帖子可能匹配到同一 sector 下多个 ticker，去重后才是真实帖子数
    g = (feat.drop_duplicates(["sector", "date", "id"])
             .groupby(["sector", "date"])
             .agg(n_posts=("id", "size"), score_sum=("score", "sum"))
             .reset_index())
    print(f"真实帖子量: 中位数 {g['n_posts'].median():.0f}, "
          f"最大 {g['n_posts'].max():,}, >150 的占比 {(g['n_posts'] > 150).mean():.1%}")

    vol = {(r.sector, r.date): torch.tensor(
                [np.log1p(r.n_posts), np.log1p(max(r.score_sum, 0.0))], dtype=torch.float32)
           for r in g.itertuples()}

    print(f"加载 day_dict: {args.day_dict}")
    dd = torch.load(args.day_dict, weights_only=False)

    hit = 0
    for k in dd:
        v = vol.get((k[0], pd.Timestamp(k[1])))
        if v is None:
            # 缓存里找不到就退回全零，而不是留着被截断的旧值
            dd[k]["day_features"] = torch.zeros(2, dtype=torch.float32)
        else:
            dd[k]["day_features"] = v
            hit += 1
    print(f"替换完成: {hit}/{len(dd)} 个 (sector, date) 命中真实帖子量")

    torch.save(dd, args.output)
    print(f"✅ 已写出 {args.output} (day_features 维度 D=2)")


if __name__ == "__main__":
    main()
