
import re
import os
import gc
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple, List
import multiprocessing
from gliner import GLiNER
from flashtext import KeywordProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json
from data_processing.scripts.labels import compute_future_realized_vol


def get_device():
    """自动选择可用设备：CUDA > MPS (Apple Silicon) > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def local_process_comment(keyword_processor, model, comment, names, comment_id, black_list, valid_tickers):
    hit_tickers = set()

    direct_hits = keyword_processor.extract_keywords(comment)
    hit_tickers.update(direct_hits)

    labels = ["company", "stock", "commercial organization"]
    entities = model.predict_entities(str(comment), labels, threshold=0.3)

    for entity in entities:
        text_span = entity['text']
        normalized_text_span = text_span.strip().upper()
        
        if normalized_text_span not in hit_tickers and normalized_text_span not in black_list:
            if normalized_text_span in names:
                matched_ticker = names.get(normalized_text_span)
                if matched_ticker and matched_ticker in valid_tickers: # Validate against original tickers
                    hit_tickers.add(matched_ticker)

    return list(hit_tickers)


def perform_local_extraction(
    df_comments_chunk,
    TEMP_RESULTS_FILE,
    valid_tickers,
    names=None,
    black_list=None,
    gliner_model_path="./gliner_model",
):
    print(f"starting processing {len(df_comments_chunk)} comments (FlashText + GLiNER + accurate name matching)...")

    if names is None:
        names = {}
    if black_list is None:
        black_list = set()

    # FlashText：直接命中 ticker
    keyword_processor = KeywordProcessor(case_sensitive=False)
    for t in valid_tickers:
        keyword_processor.add_keyword(str(t).upper(), t)

    # GLiNER：识别公司/股票实体。优先使用本地模型目录，否则从 HuggingFace 下载
    if not os.path.exists(gliner_model_path):
        gliner_model_path = "urchade/gliner_small-v2.1"
    device = get_device()
    print(f"Loading GLiNER from {gliner_model_path} on {device} ...")
    model = GLiNER.from_pretrained(gliner_model_path).to(device)
    model.eval()

    if 'combined_text' not in df_comments_chunk.columns:
        df_comments_chunk['combined_text'] = (
            df_comments_chunk['title'].fillna('') + ' ' +
            df_comments_chunk['selftext'].fillna('') + ' ' +
            df_comments_chunk['body'].fillna('')
        ).str.strip()

    # initialize lists
    all_final_results_flat = []

    total_rows = len(df_comments_chunk)
    save_interval = max(1, int(total_rows * 0.05)) # 至少 1 行，或者 10%
    last_save_row_index = 0

    df_comments_chunk['id'] = df_comments_chunk['id'].astype(str)

    with tqdm(total=total_rows, desc="Local Entity Processing (FlashText + GLiNER)") as pbar:
        for index, row in df_comments_chunk.iterrows():
            matched_tickers = local_process_comment(
                keyword_processor,
                model,
                row['combined_text'],
                names,
                row['id'],
                black_list,
                valid_tickers,
            )

            if matched_tickers:
                for ticker in matched_tickers:
                    all_final_results_flat.append({
                        "id": row['id'],
                        "source_text": row['combined_text'],
                        "matched_ticker": ticker
                    })

            pbar.update(1)

            current_row_index = pbar.n #


            if (current_row_index >= last_save_row_index + save_interval) or current_row_index == total_rows:
                df_temp = pd.DataFrame(all_final_results_flat)
                df_temp.to_csv(TEMP_RESULTS_FILE, index=False)
                print('At', current_row_index, 'rows, saved to TEMP_RESULTS_FILE')
                last_save_row_index = current_row_index
                pbar.write(f"\n⏳ At {current_row_index} / {total_rows} rows ({current_row_index/total_rows:.2%}),  mid results saved to {TEMP_RESULTS_FILE}")

    return pd.DataFrame(all_final_results_flat)

def lastNday_avg_score(
        df_text,
        window,
        score_col="score",
        ticker_col="ticker",
        date_col="date"
):
    # 1. Prepare data
    df = df_text.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    
    # 2. Aggregation: Sum scores for each ticker per day
    # This prevents the "Many-to-Many" explosion during merge
    daily_sums = (
        df.groupby([ticker_col, date_col])[score_col]
        .sum()
        .reset_index()
        .sort_values([ticker_col, date_col])
    )

    name = f"past_{window}_day_avg_score"

    # 3. Rolling Calculation: Mean of the daily sums
    daily_sums[name] = (
        daily_sums.groupby(ticker_col)[score_col]
        .shift(1)  # Look at previous days, not including today
        .rolling(window=window, min_periods=5)
        .mean()
    )

    # 4. Return unique ticker-date pairs with their scores
    # This is now a "One-to-Many" compatible dataframe
    return daily_sums[[ticker_col, date_col, name]]


#feature construction

ll = [
    # Core Financial Metrics & Reporting
    "Revenue", "Net Income", "EBITDA", "Earnings Per Share", "EPS", 
    "Profit Margin", "Free Cash Flow", "FCF", "Operating Expenses", "OpEx", 
    "Annual Report", "10-K", "Quarterly Report", "10-Q", "Current Report", 
    "8-K", "Balance Sheet", "Assets", "Liabilities", "Shareholder Equity", "Guidance",

    # Market Sentiment & Trading Jargon
    "Bullish", "Bearish", "Long", "Short", "Hedge", "Volatility", "VIX", 
    "Liquidity", "Alpha", "Beta", "Short Squeeze", "Price Action", 
    "Support and Resistance", "Breakout", "Consolidation", "Volume",

    # Valuation & Analysis
    "Market Cap", "P/E Ratio", "PEG Ratio", "Price-to-Book", "P/B", 
    "Enterprise Value", "EV", "Dividend Yield", "Payout Ratio", 
    "Return on Equity", "ROE", "Discounted Cash Flow", "DCF", 
    "Intrinsic Value", "Technical Analysis", "Fundamental Analysis",

    # Investment Vehicles & Market Structure
    "IPO", "SPAC", "ETF", "Mutual Fund", "Index Fund", "Blue Chip", 
    "Penny Stock", "Growth Stock", "Value Stock", "S&P 500", "NASDAQ", 
    "Institutional Ownership", "Float",

    # Social Media & Retail Slang
    "HODL", "Diamond Hands", "Paper Hands", "FOMO", "FUD", "Bagholder", 
    "Whale", "To the Moon", "YOLO", "Pump and Dump", "Due Diligence", "DD",

    # Macroeconomic Indicators
    "Inflation", "CPI", "Consumer Price Index", "Interest Rates", 
    "Federal Reserve", "The Fed", "Quantitative Easing", "QE", "GDP", 
    "Recession", "Soft Landing", "Hawkish", "Dovish"
]
finance_keywords = set(ll)

FLOAT_PATTERN = re.compile(
    r'(?i)\b(?:nan|inf)\b|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
)
escaped_keywords = [re.escape(k) for k in finance_keywords] #finance_keywords is a set (hash)
KEYWORD_PATTERN = re.compile(r'(?i)\b(?:' + '|'.join(escaped_keywords) + r')\b')

def count_floats(text):
    if pd.isna(text): return 0
    return len(FLOAT_PATTERN.findall(str(text)))

def count_keywords(text):
    if pd.isna(text): return 0
    return len(KEYWORD_PATTERN.findall(str(text)))

def compute_future_avg_volume( #default is 3 days
    df_stock,
    window=3,
    vol_col="VOL",
    ticker_col="ticker",
    date_col="date"
):
    """
    Compute future average trading volume over next N trading days.
    """
    df = df_stock.copy()
    df = df.sort_values([ticker_col, date_col])

    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")

    name = f"future_{window}d_avg_vol"

    df[name] = (
        df.groupby(ticker_col)[vol_col]
          .shift(-1)                               # future only
          .rolling(window, min_periods=2)
          .mean()
          .reset_index(level=0, drop=True)
    )

    return df[[ticker_col, date_col, name]]

model_path = "./finbert_model"
def load_finbert_model(model_path="./finbert_model"):
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    if device.type == "cuda":
        model = model.half()  # 半精度仅在 CUDA 上启用，MPS/CPU 用 float32
    model.eval()
    return tokenizer, model

def split_text_by_token(text, tokenizer, max_length=512):
    """seperate long text into chunks"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    segment_size = max_length - 2  
    segments = []
    for i in range(0, len(tokens), segment_size):
        segment_tokens = tokens[i:i+segment_size]
        # 转回文本（避免乱码）
        segment_text = tokenizer.decode(segment_tokens, skip_special_tokens=True)
        segments.append(segment_text)
    return segments

def get_embedding(text_list, tokenizer, model, max_length=512):
    """批量生成文本嵌入（超长文本分段+聚合）"""
    device = next(model.parameters()).device
    all_embeddings = []
    for text in text_list:
        segments = split_text_by_token(text, tokenizer, max_length)
        if not segments:
            all_embeddings.append(np.zeros(768).tolist())
            continue

        segment_inputs = tokenizer(
            segments,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        segment_inputs = {k: v.to(device) for k, v in segment_inputs.items()}
        
        with torch.no_grad():
            segment_outputs = model(**segment_inputs)
        
        segment_embeddings = segment_outputs.last_hidden_state[:, 0, :].cpu().numpy()
        final_embedding = np.mean(segment_embeddings, axis=0)  
        all_embeddings.append(final_embedding.tolist())
    
    return all_embeddings

def batch_process_embeddings_stream(
    df,
    text_col="source_text",
    output_dir="./embedding_output",
    chunk_size=5000,                 
    batch_size=256,                 
    max_length=512,
    model_path="./finbert_model"
):
    # 0. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 加载模型 (只加载一次)

    tokenizer, model = load_finbert_model(model_path)
    
    # 2. 准备数据
    df_len = len(df)
    print(f"📊 总数据量: {df_len} 行，将分为 {df_len // chunk_size + 1} 个文件保存")

    # 3. 外层循环：按 chunk_size (例如5000) 切分大块
    # range(start, stop, step)
    for start_idx in range(0, df_len, chunk_size):
        end_idx = min(start_idx + chunk_size, df_len)
        
        # 定义当前块的文件名
        file_name = f"embeddings_{start_idx}_{end_idx}.parquet"
        file_path = os.path.join(output_dir, file_name)

        # ------------------ 断点续传逻辑 ------------------
        if os.path.exists(file_path):
            print(f"⏩ 跳过已存在块: {file_name}")
            continue  # 直接进入下一轮循环
        # ------------------------------------------------

        print(f"Processing {start_idx} to {end_idx}...")
        
        # 提取当前大块的数据 (只读，不复制整个df)
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        # 确保文本列是字符串
        chunk_df[text_col] = chunk_df[text_col].fillna("").astype(str)
        
        # 容器：存放当前 chunk 的所有 embeddings
        chunk_embeddings = []

        # 4. 内层循环：按 batch_size (例如256) 进行推理
        # 使用 tqdm 显示当前 chunk 的进度
        sub_batches = range(0, len(chunk_df), batch_size)
        for i in tqdm(sub_batches, desc=f"Chunk {start_idx}", leave=False):
            sub_batch_df = chunk_df.iloc[i : i + batch_size]
            text_list = sub_batch_df[text_col].tolist()
            
            try:
                # 调用你之前的 get_embedding 函数
                embeddings = get_embedding(text_list, tokenizer, model, max_length)
                chunk_embeddings.extend(embeddings)
            except Exception as e:
                print(f"❌ Error in batch {start_idx + i}: {e}")
                # 即使报错也可以选择保存已跑的部分，或者直接抛出
                raise e

        # 5. 组装结果并保存
        # 将 embeddings 列表放入 chunk_df
        # 注意：这里我们创建一个新的轻量级 DataFrame 用于保存，避免大对象
        save_df = pd.DataFrame({
            # 保留原始索引，方便后续合并
            'original_index': chunk_df.index, 
            'embedding': chunk_embeddings
        })
        
        # 如果需要保留 ticker 或 date 等其他列，可以在这里 merge 或者是直接赋值
        # 例如: save_df['ticker'] = chunk_df['ticker'].values

        # 写入 Parquet (利用 pyarrow 引擎)
        save_df.to_parquet(file_path, engine='pyarrow', index=False)
        
        # 6. 关键步骤：强制清理内存
        del chunk_df
        del chunk_embeddings
        del save_df
        gc.collect()  # 强制运行垃圾回收
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # 清理显存碎片
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print("✅ 所有处理完成！")


#spliting data
import dask.dataframe as dd


def _to_vec(x):
    """把 embedding 列的值统一转成 1 维 numpy float32 向量"""
    return np.asarray(x, dtype=np.float32).reshape(-1)


def build_day_dict_compact(
    df: pd.DataFrame,
    L: int = 50,  
    embedding_col: str = "embedding",
    ticker_col: str = "ticker",
    date_col: str = "date",
    sort_cols: List[str] = None,
    day_num_cols: List[str] = None
) -> Tuple[Dict, int, int]:
    
    print("🚀 正在构建紧凑型 Day-Level 字典 (Memory Efficient)...")
    
    # 1. 预处理
    # 这一步如果不copy，会节省内存，但为了安全还是保留
    # 如果内存实在吃紧，可以去掉 .copy()，直接在原表操作
    df[date_col] = pd.to_datetime(df[date_col])
    
    if sort_cols is None:
        sort_cols = ["keyword_count", "word_count", "score"]
    
    if day_num_cols:
        for c in day_num_cols:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    # 获取维度
    sample_emb = _to_vec(df[embedding_col].iloc[0])
    E = sample_emb.shape[0]
    D = len(day_num_cols) + 1 if day_num_cols else 1
    
    day_dict = {}
    
    # 2. 核心优化：先排序，再分组，避免在循环里反复 sort
    # 全局排序比 200万次局部排序要快得多
    print("   ...正在全局排序...")
    df_sorted = df.sort_values([ticker_col, date_col] + sort_cols, ascending=[True, True] + [False]*len(sort_cols))
    
    # 3. 分组迭代
    print("   ...正在分组并生成 Tensor...")
    grouped = df_sorted.groupby([ticker_col, date_col], sort=False)
    
    # 使用 tqdm 显示进度，让你心里有数
    for (tic, date), group in tqdm(grouped, total=len(grouped)):
        
        # 3.1 获取原始数量 (Log1p 特征)
        raw_count = len(group)
        
        # 3.2 截断 (只取前 L 个，不补 0！)
        # 因为已经全局排好序了，直接 head(L) 即可
        curr_posts = group.head(L)
        
        # 3.3 构建紧凑 Tensor (Shape可能是 3x768, 也可能是 50x768)
        emb_matrix = np.stack(curr_posts[embedding_col].apply(_to_vec).values)
        text_tensor = torch.from_numpy(emb_matrix) # float32
        
        # 3.4 构建日频特征
        d_feats_list = []
        if day_num_cols:
            d_feats_list.extend(curr_posts[day_num_cols].iloc[0].tolist())
        d_feats_list.append(np.log1p(raw_count))
        
        item = {
            "text": text_tensor, # 变长 Tensor
            # mask 不需要存了，因为 text_tensor 的长度就是真实长度
            "day_features": torch.tensor(d_feats_list, dtype=torch.float32)
        }
            
        day_dict[(tic, date)] = item
    
    # 手动清理一下 Pandas 的垃圾
    del df_sorted
    gc.collect()
    
    print(f"✅ 字典构建完成。E={E}, D={D}")
    return day_dict, E, D


def build_time_series_samples(
    ddf: dd.DataFrame, # 传入 final_df
    day_dict: Dict,
    W: int = 20,
    label_col: str = "target",
    ticker_col: str = "ticker",
    date_col: str = "date"
) -> List[Tuple]:
    
    print(f"🚀 构建分类样本 (目标: {label_col})...")

    # 提取 Label 相关列到 Pandas（兼容 dask 和 pandas 输入）
    label_df = ddf[[ticker_col, date_col, label_col]]
    if hasattr(label_df, "compute"):
        label_df = label_df.compute()
    label_df[date_col] = pd.to_datetime(label_df[date_col])
    
    # Labels are ranked on unique stocks per date before the text join.
    if not label_df[label_col].isin([0, 1]).all():
        raise ValueError("Expected precomputed binary daily-top-20% labels")
    label_df['target'] = label_df[label_col].astype(int)
    
    # 每天每个股票只有一个 Label
    label_map = label_df.drop_duplicates([ticker_col, date_col]).set_index([ticker_col, date_col])['target'].to_dict()
    
    samples = []
    # 获取所有的 Tickers
    all_tickers = label_df[ticker_col].unique()
    
    for tic in tqdm(all_tickers, desc="处理 Tickers"):
        tic_dates = label_df[label_df[ticker_col] == tic][date_col].sort_values().unique()
        if len(tic_dates) < W: continue
            
        full_timeline = pd.bdate_range(start=tic_dates[0], end=tic_dates[-1])
        
        for idx in range(W, len(full_timeline)):
            anchor_date = full_timeline[idx]
            if (tic, anchor_date) in label_map:
                label = label_map[(tic, anchor_date)]
                lookback_dates = full_timeline[idx-W : idx]
                samples.append((tic, anchor_date, lookback_dates, label))
                
    return samples


def temporal_train_val_test_split(
    samples, 
    val_start_date: str, 
    test_start_date: str
):
    """
    slicing data based on time
    
    logic:
        Train: date < val_start_date
        Val:   val_start_date <= date < test_start_date
        Test:  date >= test_start_date
    """
    print(f"✂️ 正在进行时间切分...")
    print(f"   Train 截止: {val_start_date}")
    print(f"   Val   区间: {val_start_date} -> {test_start_date}")
    print(f"   Test  开始: {test_start_date}")

    # 转换日期格式以确保比较正确
    val_start = pd.Timestamp(val_start_date)
    test_start = pd.Timestamp(test_start_date)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    # samples 里的结构是: (tic, anchor_date, lookback_dates, label)
    # anchor_date 是索引 1
    for s in samples:
        anchor_date = s[1] 
        
        if anchor_date < val_start:
            train_samples.append(s)
        elif anchor_date < test_start:
            val_samples.append(s)
        else:
            test_samples.append(s)
            
    # --- 打印统计信息 ---
    total = len(samples)
    print("\n📊 切分结果统计:")
    print("-" * 30)
    print(f"Train Set: {len(train_samples):>7} 样本 ({len(train_samples)/total:.1%})")
    print(f"Val Set:   {len(val_samples):>7} 样本 ({len(val_samples)/total:.1%})")
    print(f"Test Set:  {len(test_samples):>7} 样本 ({len(test_samples)/total:.1%})")
    print("-" * 30)
    print(f"Total:     {total:>7} 样本")
    
    return train_samples, val_samples, test_samples

import torch
from torch.utils.data import Dataset, DataLoader

def convert_to_binary_classification(samples, threshold):
    new_samples = []
    for s in samples:
        # s[3] is Log_RV
        label = 1 if s[3] >= threshold else 0
        new_samples.append((s[0], s[1], s[2], label))
    return new_samples
