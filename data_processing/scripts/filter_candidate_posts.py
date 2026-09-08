"""
从按年份过滤好的 Reddit ndjson 中流式预筛选"可能提到股票"的候选帖子，产出 CSV。

作用:
    - 全量 ndjson 太大（百万级帖子），直接逐条跑 GLiNER 不现实。
    - 用 FlashText（基于 stocks.csv 的 ticker + 公司名）做快速关键词预筛选，
      只把包含候选关键词的帖子写给下游 pipeline，可把 GLiNER 的输入量缩小一个数量级。
    - 预筛选是宽松召回（宁可多留），精确匹配仍由 pipeline.py 里的
      FlashText 精确匹配 + GLiNER 完成。

用法:
    python data_processing/scripts/filter_candidate_posts.py \
        --input "data_processing/reddit/wallstreetbets_submissions_2021.ndjson" \
                "data_processing/reddit/wallstreetbets_submissions_2022.ndjson" \
                "data_processing/reddit/wallstreetbets_submissions_2023.ndjson" \
        --stocks "stocks.csv" \
        --output "data_processing/submissions_2021_2023_candidates.csv"

输出列: id, title, selftext, body, date, score（与 prepare_reddit_csv.py 一致）
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from flashtext import KeywordProcessor

# 允许大字段（部分 selftext/body 很长）
csv.field_size_limit(sys.maxsize)

# 与 helpers.py 保持同一套"常见英文词"判断（复用其实现，避免两处逻辑漂移）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import (
    get_common_words,
    CASHTAG_PATTERN,
    _build_short_name_map,
    _prefilter_short_name_keys,
    _COMPANY_NORM_RE,
)


def build_candidate_processor(stocks_path):
    """
    从 stocks.csv 构建候选预筛选函数。
    与 pipeline.py 第一遍匹配保持同一套规则：
      - 无歧义 ticker + 公司全称 + 非歧义公司简称：大小写不敏感
      - 多字母歧义 ticker：仅全大写
      - 单字母 ticker：仅 $TICKER
      - cashtag 对所有 ticker 生效
    预筛选是宽松召回（宁可多留），精确匹配仍由 pipeline.py 完成。
    """
    stocks = pd.read_csv(stocks_path)
    tickers = stocks['ticker'].dropna().unique()
    names = stocks['name'].dropna().unique()
    name_to_ticker = dict(zip(stocks['name'].astype(str).str.upper(), stocks['ticker']))

    common_words = get_common_words()
    clear_tickers = [t for t in tickers if len(t) > 1 and t.lower() not in common_words]
    multi_ambiguous = [t for t in tickers if len(t) > 1 and t.lower() in common_words]
    single_letters = [t for t in tickers if len(t) == 1]
    print(
        f"候选预筛选: {len(clear_tickers)} 个无歧义 ticker + {len(names)} 个公司全称(不区分大小写) | "
        f"{len(multi_ambiguous)} 个多字母歧义 ticker(仅全大写) | "
        f"{len(single_letters)} 个单字母 ticker(仅cashtag) | "
        f"歧义列表: {sorted(multi_ambiguous + single_letters)}"
    )

    short_name_map = _build_short_name_map(name_to_ticker)
    prefilter_keys = _prefilter_short_name_keys(short_name_map, common_words)
    print(f"候选预筛选额外加入公司简称: {len(prefilter_keys)} 个")

    kp = KeywordProcessor(case_sensitive=False)
    for t in clear_tickers:
        kp.add_keyword(str(t).upper())
    for n in names:
        kp.add_keyword(str(n).upper())
    for key in prefilter_keys:
        kp.add_keyword(key)

    caps_kp = KeywordProcessor(case_sensitive=True)
    for t in multi_ambiguous:
        caps_kp.add_keyword(str(t).upper())

    # 文本含标点时（AT&T、Macy's、3M...），归一化后复查一次；
    # 此时大小写信息已丢失，不能用 caps_kp（否则 can/on 等常见词全放行）
    _PUNCT_CHECK = re.compile(r"[^A-Za-z0-9 $]")

    def has_candidate(text):
        if kp.extract_keywords(text):
            return True
        if caps_kp.extract_keywords(text):
            return True
        if CASHTAG_PATTERN.search(text):
            return True
        if _PUNCT_CHECK.search(text):
            norm = _COMPANY_NORM_RE.sub(" ", text.upper())
            if kp.extract_keywords(norm):
                return True
        return False

    return has_candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, nargs='+', help='输入 ndjson 路径（可多个）')
    parser.add_argument('--stocks', default='stocks.csv', help='stocks.csv 路径（提供 ticker/name 关键词）')
    parser.add_argument('--output', required=True, help='输出候选 csv 路径')
    args = parser.parse_args()

    kp = build_candidate_processor(args.stocks)
    total = 0
    kept = 0
    with open(args.output, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=['id', 'title', 'selftext', 'body', 'date', 'score'])
        writer.writeheader()

        for path in args.input:
            print(f"处理 {path} ...")
            file_total = 0
            file_kept = 0
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    file_total += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = obj.get('created_utc')
                    if not ts:
                        continue

                    text = ' '.join(filter(None, [
                        obj.get('title') or '',
                        obj.get('selftext') or '',
                        obj.get('body') or '',
                    ])).strip()
                    if not text or not kp(text):
                        continue

                    writer.writerow({
                        'id': str(obj.get('id', '')),
                        'title': obj.get('title') or '',
                        'selftext': obj.get('selftext') or '',
                        'body': obj.get('body') or '',
                        'date': datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d'),
                        'score': obj.get('score', 0),
                    })
                    file_kept += 1

                    if file_total % 200000 == 0:
                        print(f"  ...{file_total} 行已扫描, 候选 {file_kept} ({file_kept/file_total:.1%})")

            print(f"  {path}: 扫描 {file_total}, 候选 {file_kept} ({file_kept/max(file_total,1):.1%})")
            total += file_total
            kept += file_kept

    print(f"\n完成: 共扫描 {total} 行, 候选 {kept} 行 ({kept/max(total,1):.1%})")
    print(f"已写出: {args.output}")


if __name__ == '__main__':
    main()
