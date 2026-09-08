"""
从按年份过滤好的 Reddit ndjson 中抽取 pipeline 所需的列并抽样，产出 CSV。

用法:
    python data_processing/scripts/prepare_reddit_csv.py \
        --input "data_processing/reddit/wallstreetbets_submissions_2023.ndjson" \
        --output "data_processing/2023_sample.csv" \
        --n 20000

输出列: id, title, selftext, body, date, score
- submissions 有 title/selftext，body 为空
- comments 有 body，title/selftext 为空
"""
import argparse
import json
import pandas as pd
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入 ndjson 路径')
    parser.add_argument('--output', required=True, help='输出 csv 路径')
    parser.add_argument('--n', type=int, default=20000, help='抽样条数；<=0 表示全量')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    records = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = obj.get('created_utc')
            if not ts:
                continue
            records.append({
                'id': str(obj.get('id', '')),
                'title': obj.get('title') or '',
                'selftext': obj.get('selftext') or '',
                'body': obj.get('body') or '',
                'date': datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime('%Y-%m-%d'),
                'score': obj.get('score', 0),
            })

    df = pd.DataFrame(records)
    print(f"读取 {len(df)} 条记录")

    if 0 < args.n < len(df):
        df = df.sample(n=args.n, random_state=args.seed).sort_values('date').reset_index(drop=True)
        print(f"抽样 {args.n} 条")

    df.to_csv(args.output, index=False)
    print(f"已写出: {args.output}")


if __name__ == '__main__':
    main()
