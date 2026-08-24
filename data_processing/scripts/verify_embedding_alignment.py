"""
复现 pipeline.py 里 df_unique 的行序，把已算好的 embedding 按 id 对回帖子，并严格验证对齐。

为什么需要:
    embedding_output/*.parquet 只存了 (original_index, embedding)，没有存 id。
    original_index 是 pipeline.py 中 df_unique.reset_index() 的行号，所以想复用这
    705,099 条已算好的 embedding（重跑一遍要约 5 小时），就必须一字不差地复现当时的行序。

验证方式（这是关键，不能靠"看起来对"）:
    用复现出来的顺序重新构建 L=50 的 day_dict，和现有 day_dict.pt 逐条比对张量。
    完全一致 => 行序复现正确，可以放心用来重建 L=150 / 分层采样的 day_dict。

用法:
    python data_processing/scripts/verify_embedding_alignment.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as project_config
from helpers import compute_idiosyncratic_vol

args = project_config.args
DAY_CAP = 150
SORT_COLS = ["sector", "date", "keyword_count", "word_count", "score"]
SORT_ASC = [True, True, False, False, False]


def rebuild_df_unique_order():
    """严格按 pipeline.py 的步骤复现 df_unique 的 id 顺序。"""
    feat = pd.read_parquet(os.path.join(args.data_dir, "temp_matched_features.parquet"))
    print(f"限额前匹配行数: {len(feat):,}")

    # --- pipeline.py 第 3 步：按 (sector, date) 限额 ---
    feat = feat.sort_values(SORT_COLS, ascending=SORT_ASC)
    capped = feat.groupby(["sector", "date"], sort=False).head(DAY_CAP).reset_index(drop=True)
    print(f"限额后: {len(capped):,} 行")

    # --- pipeline.py 第 1 步的 CAPM IVOL 标签，用于 inner merge 过滤日期 ---
    df_stocks = pd.read_csv(args.stocks)
    df_stocks["date"] = pd.to_datetime(df_stocks["date"]).dt.date
    df_stocks["RET"] = pd.to_numeric(df_stocks["RET"], errors="coerce")
    sector_ret = (df_stocks.groupby(["sector", "date"])["RET"].mean()
                  .reset_index().rename(columns={"RET": "sector_RET"}))

    sp = pd.read_csv(os.path.join(args.data_dir, "sp500_2020_2026.csv"))
    sp["date"] = pd.to_datetime(sp["date"]).dt.date
    sector_label = compute_idiosyncratic_vol(
        sector_ret, sp, window=5, beta_window=60,
        ret_col="sector_RET", mkt_col="SP500_RET",
        ticker_col="sector", date_col="date",
    )
    sector_label = sector_label.dropna(subset=["ivol_5"])

    # --- pipeline.py 第 4 步：inner merge 标签（pandas inner merge 保持左表键序）---
    df = pd.merge(capped, sector_label[["sector", "date", "ivol_5"]],
                  on=["sector", "date"], how="inner")
    df = df.dropna(subset=["ivol_5"])
    print(f"合并标签后: {len(df):,} 行")

    df_unique = df.drop_duplicates("id").reset_index(drop=True)
    df_unique["original_index"] = df_unique.index
    print(f"去重后待 embedding 帖子数: {len(df_unique):,}")
    return df_unique


def main():
    df_unique = rebuild_df_unique_order()

    emb = pd.read_parquet(os.path.join(args.data_dir, "embedding_output"))
    print(f"已有 embedding: {len(emb):,} 条")

    if len(df_unique) != len(emb):
        print(f"\n❌ 行数不一致 ({len(df_unique):,} vs {len(emb):,})，行序复现失败，"
              f"不能复用已有 embedding。")
        sys.exit(1)
    print("✅ 行数一致")

    merged = pd.merge(df_unique, emb, on="original_index", how="left")
    assert merged["embedding"].notna().all(), "存在未匹配上的 embedding"

    # ---- 严格验证：按 L=50 重建 day_dict，与现有文件逐张量比对 ----
    print("\n重建 L=50 的 day_dict 并与现有 day_dict.pt 比对 ...")
    old = torch.load(os.path.join(args.data_dir, "day_dict.pt"), weights_only=False)

    merged["date"] = pd.to_datetime(merged["date"])
    srt = merged.sort_values(["sector", "date", "keyword_count", "word_count", "score"],
                            ascending=[True, True, False, False, False])

    checked = mismatch = 0
    for (sec, date), grp in srt.groupby(["sector", "date"], sort=False):
        key = (sec, date)
        if key not in old:
            continue
        new_t = torch.tensor(np.stack(grp["embedding"].head(50).to_numpy()), dtype=torch.float32)
        old_t = old[key]["text"]
        if new_t.shape != old_t.shape or not torch.allclose(new_t, old_t, atol=1e-4):
            mismatch += 1
            if mismatch <= 3:
                print(f"  ✗ {key}: shape {tuple(new_t.shape)} vs {tuple(old_t.shape)}, "
                      f"maxdiff {(new_t - old_t).abs().max().item() if new_t.shape == old_t.shape else 'n/a'}")
        checked += 1

    print(f"\n比对 {checked:,} 个 (sector, date): 不一致 {mismatch}")
    if mismatch == 0:
        print("✅ 行序复现完全正确，可以复用已有 embedding 重建 day_dict")
        merged[["id", "sector", "date", "score", "keyword_count", "word_count",
                "float_count", "original_index"]].to_parquet(
            os.path.join(args.data_dir, "embedding_index.parquet"), index=False)
        print(f"   已写出 {args.data_dir}/embedding_index.parquet (id -> original_index 映射)")
    else:
        print("❌ 存在不一致，不能复用；需要重跑 embedding 并把 id 一起写出")
        sys.exit(1)


if __name__ == "__main__":
    main()
