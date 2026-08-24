"""
测试修复后的 ticker 匹配逻辑（三档规则）。

运行:
    source .venv/bin/activate
    python data_processing/scripts/test_match_fix.py

覆盖:
    A. 常见词误报消除（之前会误命中 ON/CAN/M/T/META/SHOP 等）
    B. 真实信号保留（全大写 / cashtag / 无歧义 ticker 大小写不敏感 / GLiNER 公司名）
"""
import sys
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT))

from helpers import perform_local_extraction
import config as project_config

BLACK_LIST = [
    'A', 'AI', 'AL', 'ALL', 'AN', 'AR', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CEO', 'CO', 'COO',
    'DAY', 'FL', 'FOR', 'FOUR', 'HE', 'HI', 'I', 'IT', 'LOW', 'MA', 'MD', 'MN', 'MO', 'MS',
    'MT', 'NC', 'NE', 'NEW', 'NM', 'NOW', 'NYC', 'ONE', 'ONTO', 'OR', 'OUT', 'PAY', 'SC',
    'SD', 'SEE', 'SF', 'SUN', 'TWO', 'TX', 'UP', 'USA', 'WE', 'WY', 'YOU', 'EAT', 'SEND',
    'BEST', 'HOME', 'SAFE', 'TO', 'DO', 'IN', 'OF', 'GO', 'THE', 'US', 'SO'
]

# 期望有公司名 -> ticker 映射
COMPANY_NAME_CASES = {
    "I like Macy's deals": ["M"],
    "AT&T is a dividend stock": ["T"],
    "ON Semiconductor has great margins": ["ON"],
    "Robinhood is down again": ["HOOD"],
    "Coinbase volume is up": ["COIN"],
}


def run_case(df, label):
    res = perform_local_extraction(
        df,
        os.path.join(project_config.args.data_dir, "_test_matched_chunk.csv"),
        valid_tickers=df.attrs.get('valid_tickers', DEFAULT_VALID),
        names=df.attrs.get('names', DEFAULT_NAMES),
        black_list=set(BLACK_LIST),
        gliner_model_path="./gliner_model",
        model=None,
    )
    got = {}
    for _, r in res.iterrows():
        got.setdefault(r['id'], set()).add(r['matched_ticker'])
    print(f"[{label}]")
    for _, row in df.iterrows():
        hits = got.get(row['id'], set())
        mark = "✅" if row['expected'] == hits else "❌"
        print(f"  {mark} {row['combined_text'][:60]!r} -> {sorted(hits)} (期望 {sorted(row['expected'])})")
        if row['expected'] != hits:
            return False
    return True


if __name__ == "__main__":
    import pandas as pd
    # 股票池用 config.py 指定的面板（默认 T9 上的 stocks_2020_2026.csv）
    df_stocks = pd.read_csv(project_config.args.stocks)
    my_tickers = list(df_stocks['ticker'].drop_duplicates())
    DEFAULT_VALID = [t for t in my_tickers if t not in BLACK_LIST]
    DEFAULT_NAMES = dict(zip(df_stocks['name'].astype(str).str.upper(), df_stocks['ticker']))

    # ---- A. 常见词误报消除 ----
    a = pd.DataFrame([
        {"id": "a1", "combined_text": "we will move on to the next stock", "expected": set()},
        {"id": "a2", "combined_text": "i think this can go higher", "expected": set()},
        {"id": "a3", "combined_text": "i'm going to the store later", "expected": set()},
        {"id": "a4", "combined_text": "he said it is fine with me", "expected": set()},
        {"id": "a5", "combined_text": "the best day to buy is now", "expected": set()},
        {"id": "a6", "combined_text": "the meta analysis shows no signal", "expected": set()},
        {"id": "a7", "combined_text": "we shop with our friends", "expected": set()},
        {"id": "a8", "combined_text": "this is a real coin in my pocket", "expected": set()},
        # 已知权衡：小写常见词公司名（target/meta/shop）不触发 GLiNER，
        # 需要全大写（TGT）或 $TGT 才命中；代价是丢失部分小写召回，换取不把
        # 海量含常见词的帖子送入 GLiNER。
        {"id": "a9", "combined_text": "she shops at target for the dividend yield", "expected": set()},
    ])
    ok_a = run_case(a, "A. 常见词误报消除")

    # ---- B. 真实信号保留 ----
    b = pd.DataFrame([
        {"id": "b1", "combined_text": "ON is the best semi stock", "expected": {"ON"}},
        {"id": "b2", "combined_text": "$ON earnings today are huge", "expected": {"ON"}},
        {"id": "b3", "combined_text": "I bought $M stock yesterday", "expected": {"M"}},
        {"id": "b4", "combined_text": "CAN is undervalued right now", "expected": {"CAN"}},
        {"id": "b5", "combined_text": "GME to the moon baby", "expected": {"GME"}},
        {"id": "b6", "combined_text": "tsla is a buy", "expected": {"TSLA"}},
        {"id": "b7", "combined_text": "META earnings were great", "expected": {"META"}},
        {"id": "b8", "combined_text": "$T is my dividend pick", "expected": {"T"}},
        {"id": "b9", "combined_text": "$HOOD and $COIN both pumping", "expected": {"HOOD", "COIN"}},
        {"id": "b10", "combined_text": "$TSLA $GME $PLTR", "expected": {"TSLA", "GME", "PLTR"}},
    ])
    ok_b = run_case(b, "B. 真实信号保留(直接匹配)")

    # ---- C. GLiNER 公司名兜底 ----
    c = pd.DataFrame([
        {"id": cid, "combined_text": text, "expected": set(exp)}
        for cid, (text, exp) in enumerate(COMPANY_NAME_CASES.items())
    ])
    ok_c = run_case(c, "C. GLiNER 公司名兜底")

    print("\n" + "=" * 50)
    print(f"A. 常见词误报消除: {'PASS' if ok_a else 'FAIL'}")
    print(f"B. 真实信号保留:   {'PASS' if ok_b else 'FAIL'}")
    print(f"C. GLiNER 兜底:    {'PASS' if ok_c else 'FAIL'}")
    sys.exit(0 if (ok_a and ok_b and ok_c) else 1)
