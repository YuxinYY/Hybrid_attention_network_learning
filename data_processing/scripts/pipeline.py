import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import torch

# 将项目根目录加入 sys.path，以便从根目录导入 config
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from data_processing.scripts.labels import LABEL_SCHEME, daily_top_fraction_labels, relabel_samples
args = config.args
from helpers import (
    compute_future_realized_vol, perform_local_extraction, count_floats,
    count_keywords, batch_process_embeddings_stream, build_day_dict_compact,
    build_time_series_samples, temporal_train_val_test_split,
)

# ---------- 1. 加载数据 ----------
# loading social media text
df_comments = pd.read_csv(args.submissionandcomments_dir)
df_comments['id'] = df_comments['id'].astype(str)

df_stocks = pd.read_csv(args.stocks)  # 列: date, ticker, name, RET, VOL
df_stocks['date'] = pd.to_datetime(df_stocks['date'])

rv5 = compute_future_realized_vol(df_stocks, window=5)  # future 5-day realized volatility
daily_labels = daily_top_fraction_labels(rv5)

my_tickers = list(df_stocks['ticker'].drop_duplicates())  # the list of stocks we should care about

# filter
black_list = [
    'A', 'AI', 'AL', 'ALL', 'AN', 'AR', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CEO', 'CO', 'COO', 'DAY', 'FL', 'FOR', 'FOUR', 'HE', 'HI', 'I', 'IT', 'LOW', 'MA', 'MD', 'MN', 'MO', 'MS', 'MT', 'NC', 'NE', 'NEW', 'NM', 'NOW', 'NYC', 'ONE', 'ONTO', 'OR', 'OUT', 'PAY', 'SC', 'SD', 'SEE', 'SF', 'SUN', 'TWO', 'TX', 'UP', 'USA', 'WE', 'WY', 'YOU', 'EAT', 'PAY', 'SEE', 'SEND', 'BEST', 'HOME', 'LOW', 'SAFE', 'OR', 'TO', 'DO', 'IN', 'OF', 'GO', 'THE', 'BY', 'US', 'NOW', 'SO', 'UP'

]
filtered_tickers = [t for t in my_tickers if t not in black_list]

# 公司名(大写) -> ticker 映射，供 GLiNER 实体匹配
names = dict(zip(df_stocks['name'].astype(str).str.upper(), df_stocks['ticker']))

# ---------- 2. FlashText + GLiNER 匹配 ----------
TEMP_RESULTS_FILE = "./data_processing/temp_matched.csv"

matched_df = perform_local_extraction(
    df_comments,
    TEMP_RESULTS_FILE,
    valid_tickers=filtered_tickers,
    names=names,
    black_list=set(black_list),
    gliner_model_path="./gliner_model",
)

# merging matched results with stock volatility
matched_df = matched_df[matched_df['matched_ticker'].isin(filtered_tickers)]

df = pd.merge(matched_df, df_comments[['id', 'date', 'score']], on=['id'])  # score is the net upvotes from Reddit
df['date'] = pd.to_datetime(df['date'])
df.rename(columns={"matched_ticker": "ticker"}, inplace=True)
df = pd.merge(df, df_stocks[['ticker', 'date', 'VOL']], on=['date', 'ticker'], how='inner')

# structured features
df['float_count'] = df['source_text'].apply(count_floats)  # df has a "source_text" column after matching
df['keyword_count'] = df['source_text'].apply(count_keywords)
df['word_count'] = df['source_text'].fillna('').apply(lambda x: len(str(x).split()))

# computing labels: realized volatility
df = pd.merge(df, daily_labels, on=['ticker', 'date'], how='inner', validate='many_to_one')
df = df.dropna(subset=['RV_5'])

print(f"匹配后样本量: {len(df)} 行, 涉及 {df['ticker'].nunique()} 个 ticker")

# ---------- 3. FinBERT embedding ----------
embedding_dir = "./data_processing/embedding_output_daily_top20"
df = df.reset_index(drop=True)
batch_process_embeddings_stream(  # this is saved using parquet
    df=df,
    output_dir=embedding_dir,
    chunk_size=10000,
    batch_size=256,
    model_path="./finbert_model",
)

df_result = pd.read_parquet(embedding_dir)  # 2 columns: original_index and embedding
df = df.reset_index(drop=True)
df['original_index'] = df.index
final_df = pd.merge(df, df_result, on='original_index', how='left')

# ---------- 4. 构建样本并切分 ----------
day_dict, E, D = build_day_dict_compact(final_df)
samples = build_time_series_samples(final_df, day_dict, W=20, label_col="target")
train_s, val_s, test_s = temporal_train_val_test_split(samples, "2023-08-01", "2023-11-01")
train_s = relabel_samples(train_s, daily_labels, boundary="2023-08-01")
val_s = relabel_samples(val_s, daily_labels, boundary="2023-11-01")

# ---------- 5. 保存 ----------
save_dir = "./data_processing"
os.makedirs(save_dir, exist_ok=True)
print(f"Saving data to {save_dir} ...")
torch.save(day_dict, os.path.join(save_dir, "day_dict.pt"))
torch.save(train_s, os.path.join(save_dir, "train_samples.pt"))
torch.save(val_s, os.path.join(save_dir, "val_samples.pt"))
torch.save(test_s, os.path.join(save_dir, "test_samples.pt"))
config_dict = {"E": E, "D": D, "L": 50, "label_scheme": LABEL_SCHEME,
               "stocks": args.stocks, "val_start": "2023-08-01", "test_start": "2023-11-01"}
torch.save(config_dict, os.path.join(save_dir, "config.pt"))
print("✅ Pipeline 完成")
