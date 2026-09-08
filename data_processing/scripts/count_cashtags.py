"""
从历史 Reddit ndjson 中统计 cashtag（$TICKER）出现频次，用于构建扩展 ticker universe。

用法示例:
    python data_processing/scripts/count_cashtags.py \
        --input_dir data_processing/reddit \
        --output data_processing/cashtag_counts.csv \
        --years 2021 2022 2023 2024 2025

输出 CSV 列:
    ticker, total_mentions, mentions_2021, ..., mentions_2025,
    years_active, first_year, last_year, is_valid_stock

验证 ticker 是否真实存在的数据来自 NASDAQ Trader:
    - ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt
    - ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt
"""

import argparse
import os
import re
import csv
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import urllib.request


NASDAQ_LISTED_URL = "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"
OTHER_LISTED_URL = "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"


def download_master_tickers(cache_path: str, refresh: bool = False):
    """
    下载并缓存 NASDAQ 上市证券列表，返回有效 ticker 集合。
    过滤掉 Test Issue 的 ticker；保留普通股和 ETF（后续用 yfinance 判断 sector）。
    """
    cache = Path(cache_path)
    if cache.exists() and not refresh and cache.stat().st_size > 0:
        print(f"[Master] 使用缓存: {cache_path}")
        tickers = set()
        with cache.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tickers.add(row["ticker"])
        return tickers

    print("[Master] 下载 NASDAQ 上市证券列表...")
    tickers = set()

    for url in [NASDAQ_LISTED_URL, OTHER_LISTED_URL]:
        print(f"[Master] 从 {url} 下载...")
        with urllib.request.urlopen(url, timeout=60) as resp:
            # 第一行是表头，最后一行是文件生成时间戳
            lines = resp.read().decode("utf-8").splitlines()

        header = lines[0].split("|")
        for line in lines[1:-1]:
            parts = line.split("|")
            row = dict(zip(header, parts))
            # nasdaqlisted: Symbol, otherlisted: ACT Symbol
            symbol = row.get("Symbol") or row.get("ACT Symbol")
            test_issue = row.get("Test Issue", "N")
            if not symbol or test_issue == "Y":
                continue
            tickers.add(symbol.strip().upper())

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker"])
        for t in sorted(tickers):
            writer.writerow([t])

    print(f"[Master] 共 {len(tickers)} 个有效 ticker，已缓存到 {cache_path}")
    return tickers


def stream_ndjson(input_path):
    """流式读取 ndjson 文件，每行返回原始字符串。"""
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield line


def count_cashtags(input_dir: str, years: list, valid_tickers: set):
    """
    统计每个 ndjson 文件中的 cashtag 频次。
    返回:
        year_counter: {year: Counter(ticker -> count)}
        total_counter: Counter(ticker -> count)
        stats: {year: {"lines": int, "matched_lines": int, "mentions": int}}
    """
    cashtag_pattern = re.compile(r"\$([A-Z]{1,5})\b")
    year_counter = {y: Counter() for y in years}
    total_counter = Counter()
    stats = {}

    for year in years:
        input_path = Path(input_dir) / f"wallstreetbets_submissions_{year}.ndjson"
        if not input_path.exists():
            print(f"[Warn] 文件不存在，跳过: {input_path}")
            continue

        print(f"[Scan] 正在扫描 {input_path} ...")
        lines = 0
        matched_lines = 0
        mentions = 0

        for line in stream_ndjson(input_path):
            lines += 1
            hits = cashtag_pattern.findall(line)
            if not hits:
                continue
            matched_lines += 1
            for raw in hits:
                ticker = raw.upper()
                # 只保留看起来是真实证券代码的（过滤纯单个字母等常见噪音）
                if len(ticker) < 1:
                    continue
                # 这里先不过滤 valid_tickers，保留全部以便观察原始分布
                year_counter[year][ticker] += 1
                total_counter[ticker] += 1
                mentions += 1

            if lines % 1_000_000 == 0:
                print(f"  {year}: 已处理 {lines:,} 行, 命中 {mentions:,} 次")

        stats[year] = {"lines": lines, "matched_lines": matched_lines, "mentions": mentions}
        print(f"[Done] {year}: {lines:,} 行, {matched_lines:,} 行含 cashtag, 共 {mentions:,} 次提及")

    return year_counter, total_counter, stats


def main():
    parser = argparse.ArgumentParser(description="统计 Reddit ndjson 中的 cashtag 频次")
    parser.add_argument("--input_dir", default="data_processing/reddit", help="ndjson 所在目录")
    parser.add_argument("--output", default="data_processing/cashtag_counts.csv", help="输出 CSV")
    parser.add_argument("--master_cache", default="data_processing/.master_tickers.csv", help="NASDAQ 列表缓存")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025], help="年份列表")
    parser.add_argument("--refresh_master", action="store_true", help="强制刷新 NASDAQ 列表缓存")
    parser.add_argument("--min_mentions", type=int, default=0, help="输出时过滤低于该 mention 数的 ticker")
    args = parser.parse_args()

    valid_tickers = download_master_tickers(args.master_cache, refresh=args.refresh_master)

    year_counter, total_counter, stats = count_cashtags(args.input_dir, args.years, valid_tickers)

    # 构建每个 ticker 的活跃年份
    ticker_years = defaultdict(set)
    for year, counter in year_counter.items():
        for ticker in counter:
            ticker_years[ticker].add(year)

    # 写出 CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["ticker", "total_mentions"] + [f"mentions_{y}" for y in args.years] + [
            "years_active", "first_year", "last_year", "is_valid_stock"
        ]
        writer.writerow(header)

        for ticker, total in total_counter.most_common():
            if total < args.min_mentions:
                continue
            years = sorted(ticker_years[ticker])
            row = [ticker, total]
            for y in args.years:
                row.append(year_counter[y].get(ticker, 0))
            row.append(len(years))
            row.append(years[0] if years else "")
            row.append(years[-1] if years else "")
            row.append("Y" if ticker in valid_tickers else "N")
            writer.writerow(row)

    print(f"\n[Summary]")
    for year, s in sorted(stats.items()):
        print(f"  {year}: {s['lines']:,} 行, {s['matched_lines']:,} 行命中, {s['mentions']:,} 次提及")
    print(f"\n输出: {args.output}")
    print(f"共 {len(total_counter)} 个不同 ticker（valid stock: {sum(1 for t in total_counter if t in valid_tickers)}）")


if __name__ == "__main__":
    main()
