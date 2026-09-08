"""
构建 (sector, date) 级别的统计量特征表，替代 post-level attention。

动机:
    原来的做法是每个 sector-day 取 50 条帖子的 768 维 embedding 送进 post-level attention。
    问题有三个:
      1. 中位数 340 条、最多 24.8 万条的日子只留 50 条，且筛选规则是"关键词最密集+最长"，
         系统性偏向长 DD 帖，不具代表性；
      2. 输入 (20, 50, 768) 让 GRU 的参数量涨到 44 万，而训练样本只有 4,640 个（95:1）；
      3. 768 维向量取平均后金融语义基本被洗掉（实测 mean-embedding AUC 0.505）。
    改成日级统计量后，特征在**当天全部有 embedding 的帖子**上计算（限额后最多 150 条，
    是原来 50 条的 3 倍），而且量纲和含义都可解释。

情绪从哪来:
    项目里的 ./finbert_model 只存了 BertModel（base encoder），没有情绪分类头，
    所以此前用的一直是 [CLS] 向量，从没用上 FinBERT 的金融情绪能力。
    FinBERT 的分类头是 classifier(tanh(pooler_dense(CLS)))，而我们恰好存了 CLS，
    所以补上分类头权重后可以**离线**算出全部 705k 条帖子的情绪，不必重跑推理。
    分类头由 fetch 脚本存在 finbert_model/classifier_head.pt（2,307 个参数）。

输出 <data_dir>/day_features.parquet（data_dir 由 config.py 指定，默认 T9），
每个 (sector, date) 一行:
    情绪类  neg_mean, pos_mean, neu_mean, net_mean, net_std,
            net_q10, net_q25, net_q50, net_q75, net_q90,
            frac_strong_neg, frac_strong_pos, frac_opinionated
    关注类  log_n_posts, log_score_sum, score_mean, log_score_max,
            frac_viral, n_tickers, n_embedded

用法:
    python data_processing/scripts/build_day_features.py
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

from verify_embedding_alignment import rebuild_df_unique_order
import config as project_config

DATA = project_config.args.data_dir
EMB_DIR = Path(DATA) / "embedding_output"
FEATURES = os.path.join(DATA, "temp_matched_features.parquet")
HEAD = "finbert_model/classifier_head.pt"
OUT = os.path.join(DATA, "day_features.parquet")


def compute_sentiment():
    """对全部已存 CLS embedding 离线推断 FinBERT 情绪，返回 original_index -> (pos, neg, neu)。"""
    from transformers import AutoModel

    print("加载 FinBERT pooler 与分类头 ...")
    base = AutoModel.from_pretrained("./finbert_model").eval()
    head = torch.load(HEAD, weights_only=False)
    W, b = head["weight"], head["bias"]

    files = sorted(EMB_DIR.glob("*.parquet"))
    print(f"逐块推断情绪，共 {len(files)} 个 chunk ...")
    idxs, probs = [], []
    with torch.no_grad():
        for i, f in enumerate(files):
            df = pd.read_parquet(f)
            cls = torch.tensor(np.stack(df["embedding"].to_numpy()), dtype=torch.float32)
            pooled = torch.tanh(base.pooler.dense(cls))
            p = torch.softmax(pooled @ W.T + b, dim=-1)  # 列序: positive, negative, neutral
            idxs.append(df["original_index"].to_numpy())
            probs.append(p.numpy().astype(np.float32))
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(files)}")

    out = pd.DataFrame(np.concatenate(probs), columns=["pos", "neg", "neu"])
    out["original_index"] = np.concatenate(idxs)
    print(f"情绪推断完成: {len(out):,} 条帖子")
    print(f"  平均 pos={out['pos'].mean():.3f} neg={out['neg'].mean():.3f} neu={out['neu'].mean():.3f}")
    return out


def sentiment_day_stats(sent):
    """按 (sector, date) 聚合情绪分布。"""
    sent["net"] = sent["pos"] - sent["neg"]          # 净情绪，取值 [-1, 1]
    sent["strong_neg"] = (sent["neg"] > 0.8).astype(np.float32)
    sent["strong_pos"] = (sent["pos"] > 0.8).astype(np.float32)
    sent["opinionated"] = (sent["neu"] < 0.5).astype(np.float32)  # 有明确观点（非中性）

    g = sent.groupby(["sector", "date"])
    stats = g.agg(
        neg_mean=("neg", "mean"),
        pos_mean=("pos", "mean"),
        neu_mean=("neu", "mean"),
        net_mean=("net", "mean"),
        net_std=("net", "std"),                       # 分歧度
        frac_strong_neg=("strong_neg", "mean"),
        frac_strong_pos=("strong_pos", "mean"),
        frac_opinionated=("opinionated", "mean"),
        n_embedded=("net", "size"),
    ).reset_index()

    q = g["net"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    q.columns = [f"net_q{int(c*100)}" for c in q.columns]
    stats = stats.merge(q.reset_index(), on=["sector", "date"])
    stats["net_std"] = stats["net_std"].fillna(0.0)
    return stats


def attention_day_stats():
    """按 (sector, date) 聚合关注度，用限额**之前**的全量匹配结果。"""
    print(f"\n读取限额前匹配结果 {FEATURES} ...")
    feat = pd.read_parquet(FEATURES, columns=["id", "date", "sector", "score", "ticker"])
    feat["date"] = pd.to_datetime(feat["date"])
    print(f"  {len(feat):,} 行")

    # 一条帖子可能匹配同 sector 下多个 ticker，帖子量要按 id 去重
    posts = feat.drop_duplicates(["sector", "date", "id"])
    g = posts.groupby(["sector", "date"])
    out = g.agg(
        n_posts=("id", "size"),
        score_sum=("score", "sum"),
        score_mean=("score", "mean"),
        score_max=("score", "max"),
    ).reset_index()
    out["frac_viral"] = (g["score"].apply(lambda s: (s > 100).mean())).to_numpy()
    out["n_tickers"] = feat.groupby(["sector", "date"])["ticker"].nunique().to_numpy()

    out["log_n_posts"] = np.log1p(out["n_posts"])
    out["log_score_sum"] = np.log1p(out["score_sum"].clip(lower=0))
    out["log_score_max"] = np.log1p(out["score_max"].clip(lower=0))
    return out.drop(columns=["score_sum", "score_max"])


def main():
    # id / sector / date 与 original_index 的对应关系（行序复现已由 verify 脚本验证）
    print("复现 embedding 行序 ...")
    idx = rebuild_df_unique_order()[["id", "sector", "date", "original_index"]]
    idx["date"] = pd.to_datetime(idx["date"])

    sent = compute_sentiment()
    sent = idx.merge(sent, on="original_index", how="left")
    assert sent["neg"].notna().all(), "存在未对上情绪的帖子"

    s_stats = sentiment_day_stats(sent)
    a_stats = attention_day_stats()

    day = a_stats.merge(s_stats, on=["sector", "date"], how="inner")
    day = day.sort_values(["sector", "date"]).reset_index(drop=True)
    day.to_parquet(OUT, index=False)

    feat_cols = [c for c in day.columns if c not in ("sector", "date")]
    print(f"\n✅ 已写出 {OUT}: {len(day):,} 个 (sector, date), {len(feat_cols)} 个特征")
    print(f"   {feat_cols}")
    print("\n各 sector 平均净情绪 / 日均帖子量:")
    print(day.groupby("sector")[["net_mean", "n_posts", "frac_strong_neg"]].mean().round(3))


if __name__ == "__main__":
    main()
