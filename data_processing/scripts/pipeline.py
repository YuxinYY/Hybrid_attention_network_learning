"""
从候选 Reddit CSV 生成 HAN 训练所需的 day_dict.pt + config.pt。

作用:
    读取 submissions/comments 候选 CSV（已由 filter_candidate_posts.py 预筛选），
    经过 FlashText + GLiNER 匹配 ticker → FinBERT embedding → sector 级聚合，
    最终输出 day_dict.pt 和 config.pt。

用法:
    python data_processing/scripts/pipeline.py \
        --candidate_csvs \
            /Volumes/T9/.../submissions_2021_2023_candidates.csv,\
            /Volumes/T9/.../comments_2021_2023_candidates.csv,\
            /Volumes/T9/.../submissions_2024_2025_candidates.csv,\
            /Volumes/T9/.../comments_2024_2025_candidates.csv \
        --stocks /Volumes/T9/.../stocks_2020_2026.csv \
        --output_dir /Volumes/T9/.../data_processing \
        --finbert_model ./finbert_model \
        --gliner_model ./gliner_model

输出:
    {output_dir}/day_dict.pt
    {output_dir}/config.pt
    {output_dir}/matched_all.parquet   (可选，保留匹配结果供复用)
    {output_dir}/embedding_output/     (parquet 分块 embedding)
"""
import argparse
import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import (
    batch_process_embeddings_stream,
    build_day_dict_compact,
    count_floats,
    count_keywords,
    perform_local_extraction,
)


def load_stocks(path):
    """加载股票面板，返回 ticker -> sector 映射和公司名映射。"""
    df = pd.read_csv(path)
    df = df.dropna(subset=['ticker', 'sector'])
    df['ticker'] = df['ticker'].astype(str).str.upper()
    df['name'] = df['name'].astype(str).str.strip()
    ticker_to_sector = dict(zip(df['ticker'], df['sector']))
    # name -> ticker: 用第一个非空名字
    names = {}
    for _, row in df.iterrows():
        name = str(row['name']).strip()
        if name and name.upper() != 'NAN':
            names[name.upper()] = row['ticker']
    return ticker_to_sector, names, sorted(df['ticker'].unique())


def read_candidates_in_chunks(paths, chunksize=200_000):
    """逐个文件、逐块读取候选 CSV，yield DataFrame chunks。"""
    for p in paths:
        print(f"\n📂 读取候选文件: {p}")
        for chunk in pd.read_csv(p, chunksize=chunksize):
            yield chunk


def load_gliner(gliner_model_path):
    """预加载 GLiNER 模型，避免每个 chunk 重复加载。"""
    from gliner import GLiNER
    from helpers import get_device
    if not os.path.exists(gliner_model_path):
        gliner_model_path = "urchade/gliner_small-v2.1"
    device = get_device()
    print(f"Loading GLiNER from {gliner_model_path} on {device} ...")
    model = GLiNER.from_pretrained(gliner_model_path).to(device)
    model.eval()
    return model


def process_chunk(chunk, valid_tickers, names, gliner_model, temp_file):
    """对一块候选数据做 ticker 匹配，返回匹配到的 DataFrame（保留 date, score）。"""
    # 确保列存在
    for col in ['id', 'title', 'selftext', 'body', 'date', 'score']:
        if col not in chunk.columns:
            chunk[col] = None

    # 合并文本
    chunk['combined_text'] = (
        chunk['title'].fillna('') + ' ' +
        chunk['selftext'].fillna('') + ' ' +
        chunk['body'].fillna('')
    ).str.strip()

    # 空文本直接跳过
    chunk = chunk[chunk['combined_text'].str.len() > 0].copy()
    if len(chunk) == 0:
        return pd.DataFrame()

    matched = perform_local_extraction(
        chunk,
        temp_file,
        valid_tickers=valid_tickers,
        names=names,
        black_list=set(),  # 如需黑名单可后续补充
        gliner_model_path="./gliner_model",  # 仅当 model 为 None 时备用
        model=gliner_model,
    )

    if len(matched) == 0:
        return pd.DataFrame()

    # 把 date 和 score 从原始 chunk 合并回来
    keep_cols = ['id', 'date', 'score']
    matched = matched.merge(chunk[keep_cols], on='id', how='left')
    return matched


def main():
    parser = argparse.ArgumentParser(description="Reddit 候选 → HAN day_dict 数据管道")
    parser.add_argument('--candidate_csvs', required=True,
                        help='逗号分隔的候选 CSV 路径（submissions + comments）')
    parser.add_argument('--stocks', required=True,
                        help='股票面板 CSV，含 ticker, name, sector 列')
    parser.add_argument('--output_dir', required=True,
                        help='输出目录（存 day_dict.pt, config.pt 等）')
    parser.add_argument('--finbert_model', default='./finbert_model',
                        help='本地 FinBERT 模型目录')
    parser.add_argument('--gliner_model', default='./gliner_model',
                        help='本地 GLiNER 模型目录')
    parser.add_argument('--chunksize', type=int, default=200_000,
                        help='读取 CSV 的块大小')
    parser.add_argument('--L', type=int, default=50,
                        help='每天每个 sector 保留的最大帖子数')
    parser.add_argument('--save_matched', action='store_true',
                        help='是否保存匹配后的中间表 matched_all.parquet')
    parser.add_argument('--base_day_dict', default=None,
                        help='已有的 day_dict.pt 路径；生成后会与该字典合并（用于追加 2024-2025 数据）')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    temp_file = os.path.join(args.output_dir, 'temp_matched.csv')
    embedding_dir = os.path.join(args.output_dir, 'embedding_output')

    # 1. 加载股票面板
    print(f"📊 加载股票面板: {args.stocks}")
    ticker_to_sector, names, valid_tickers = load_stocks(args.stocks)
    print(f"   tickers: {len(valid_tickers)}, sectors: {len(set(ticker_to_sector.values()))}")

    # 2. 预加载 GLiNER（只加载一次）
    gliner_model = load_gliner(args.gliner_model)

    # 3. 分块匹配
    candidate_paths = [p.strip() for p in args.candidate_csvs.split(',')]
    all_matched_parts = []
    part_idx = 0

    for chunk in read_candidates_in_chunks(candidate_paths, chunksize=args.chunksize):
        print(f"\n🔍 处理 chunk {part_idx}, 行数 {len(chunk)}")
        matched = process_chunk(chunk, valid_tickers, names, gliner_model, temp_file)
        print(f"   匹配到 {len(matched)} 行")

        if len(matched) > 0:
            # 保存当前块，避免内存累积
            part_path = os.path.join(args.output_dir, f'matched_part_{part_idx:04d}.parquet')
            matched.to_parquet(part_path, index=False)
            all_matched_parts.append(part_path)
            del matched

        del chunk
        gc.collect()
        part_idx += 1

    if not all_matched_parts:
        print("❌ 没有匹配到任何帖子，退出")
        return

    # 3. 合并匹配结果并映射到 sector
    print(f"\n🧩 合并 {len(all_matched_parts)} 个匹配分块...")
    matched_dfs = [pd.read_parquet(p) for p in all_matched_parts]
    df_matched = pd.concat(matched_dfs, ignore_index=True)
    for p in all_matched_parts:
        os.remove(p)
    del matched_dfs
    gc.collect()

    # ticker -> sector
    df_matched['ticker'] = df_matched['matched_ticker'].str.upper()
    df_matched['sector'] = df_matched['ticker'].map(ticker_to_sector)
    df_matched = df_matched.dropna(subset=['sector'])
    print(f"   映射到 sector 后: {len(df_matched)} 行, {df_matched['sector'].nunique()} 个 sector")

    # 日期标准化
    df_matched['date'] = pd.to_datetime(df_matched['date'], errors='coerce')
    df_matched = df_matched.dropna(subset=['date'])
    print(f"   有效日期后: {len(df_matched)} 行")

    # 文本特征
    df_matched['float_count'] = df_matched['source_text'].apply(count_floats)
    df_matched['keyword_count'] = df_matched['source_text'].apply(count_keywords)
    df_matched['word_count'] = df_matched['source_text'].fillna('').apply(lambda x: len(str(x).split()))

    if args.save_matched:
        matched_path = os.path.join(args.output_dir, 'matched_all.parquet')
        df_matched.to_parquet(matched_path, index=False)
        print(f"   已保存: {matched_path}")

    # 4. FinBERT embedding
    print(f"\n🤖 生成 FinBERT embedding...")
    batch_process_embeddings_stream(
        df_matched,
        text_col='source_text',
        output_dir=embedding_dir,
        chunk_size=5000,
        batch_size=256,
        max_length=512,
        model_path=args.finbert_model,
    )

    # 5. 合并 embedding
    print(f"\n🔗 合并 embedding...")
    emb_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('.parquet')])
    emb_dfs = [pd.read_parquet(os.path.join(embedding_dir, f)) for f in emb_files]
    df_emb = pd.concat(emb_dfs, ignore_index=True)
    df_matched = df_matched.reset_index(drop=True)
    df_matched['original_index'] = df_matched.index
    final_df = pd.merge(df_matched, df_emb, on='original_index', how='inner')
    print(f"   合并后: {len(final_df)} 行")
    del df_emb, df_matched
    gc.collect()

    # 6. 构建 day_dict (sector 级)
    print(f"\n📅 构建 day_dict (L={args.L})...")
    day_dict, E, D = build_day_dict_compact(
        final_df,
        L=args.L,
        embedding_col='embedding',
        ticker_col='sector',
        date_col='date',
        sort_cols=['keyword_count', 'word_count', 'score'],
        day_num_cols=None,
    )

    # 7. 与已有 day_dict 合并（如果提供）
    if args.base_day_dict and os.path.exists(args.base_day_dict):
        print(f"\n🔄 合并已有 day_dict: {args.base_day_dict}")
        base_day_dict = torch.load(args.base_day_dict, weights_only=False)
        n_before = len(day_dict)
        day_dict.update(base_day_dict)
        print(f"   合并后: {n_before} -> {len(day_dict)} 个 (sector, date) 条目")
        del base_day_dict
        gc.collect()

    # 8. 保存
    day_dict_path = os.path.join(args.output_dir, 'day_dict.pt')
    config_path = os.path.join(args.output_dir, 'config.pt')
    torch.save(day_dict, day_dict_path)
    torch.save({'E': E, 'D': D, 'L': args.L}, config_path)
    print(f"\n✅ 完成")
    print(f"   day_dict: {day_dict_path} ({len(day_dict)} 个 (sector, date) 条目)")
    print(f"   config: {config_path} (E={E}, D={D}, L={args.L})")


if __name__ == '__main__':
    main()
