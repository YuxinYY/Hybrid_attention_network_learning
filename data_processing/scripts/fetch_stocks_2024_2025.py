"""
抓取 2024/2025 及周边年份的股票日线数据，用于扩展 stocks 面板。

背景:
    - 现有 stocks_2020_2023.csv 只覆盖到 2023-12-29。
    - 为了给 2024/2025 构建 sector IVOL 标签，需要:
        * 2024-01-01 之前约 250 个交易日的历史（trailing beta 窗口）
        * 2025-12-31 之后约 20 个交易日（未来残差波动窗口）
      因此默认抓取 2022-01-01 ~ 2026-02-01。
    - name / sector 列沿用 stocks_2020_2023.csv 的原始映射，保证与
      filter_candidate_posts.py / pipeline.py 的公司名匹配逻辑一致。

输出列（与 stocks_2020_2023.csv 一致）:
    date, ticker, name, sector, RET, VOL

用法:
    python data_processing/scripts/fetch_stocks_2024_2025.py \
        --universe data_processing/stocks_2020_2023.csv \
        --output /Volumes/T9/.../data_processing/stocks_2022_2026.csv \
        --index_output /Volumes/T9/.../data_processing/sp500_2022_2026.csv
"""
import argparse
import time

import pandas as pd
import yfinance as yf


def load_universe(universe_path: str):
    """从现有 stocks.csv 读取 (ticker, name, sector) 唯一映射。"""
    df = pd.read_csv(universe_path)
    df = df.dropna(subset=['ticker'])
    # 每个 ticker 的 name/sector 唯一（已验证），取最后一组
    mapping = (df.sort_values('date')
                 .groupby('ticker')[['name', 'sector']]
                 .last()
                 .reset_index())
    print(f"universe: {len(mapping)} 个 ticker")
    return list(mapping.itertuples(index=False, name=None))


def fetch_ticker(ticker: str, start: str, end: str):
    """下载单只股票日线，返回 date, RET, VOL；失败返回 None。"""
    hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return None
    hist = hist.reset_index()
    hist['RET'] = hist['Close'].pct_change()
    hist = hist.dropna(subset=['RET'])
    df = pd.DataFrame({
        'date': hist['Date'].dt.strftime('%Y-%m-%d'),
        'RET': hist['RET'],
        'VOL': hist['Volume'],
    })
    return df


def fetch_index(start: str, end: str):
    """下载 ^GSPC，返回 date, SP500_RET。"""
    hist = yf.Ticker("^GSPC").history(start=start, end=end, auto_adjust=True)
    hist = hist.reset_index()
    hist['SP500_RET'] = hist['Close'].pct_change()
    hist = hist.dropna(subset=['SP500_RET'])
    df = pd.DataFrame({
        'date': hist['Date'].dt.strftime('%Y-%m-%d'),
        'SP500_RET': hist['SP500_RET'],
    })
    return df


def main():
    p = argparse.ArgumentParser(description="抓取 2024/2025 股票日线扩展数据")
    p.add_argument('--start', default='2022-01-01')
    p.add_argument('--end', default='2026-02-01')
    p.add_argument('--universe', default='data_processing/stocks_2020_2023.csv')
    p.add_argument('--output', default='stocks_2022_2026.csv')
    p.add_argument('--index_output', default='sp500_2022_2026.csv')
    p.add_argument('--sleep', type=float, default=0.12)
    args = p.parse_args()

    # 1. 指数
    print(f"[1/2] 抓取 ^GSPC {args.start} ~ {args.end} ...")
    idx = fetch_index(args.start, args.end)
    idx.to_csv(args.index_output, index=False)
    print(f"  ✅ {args.index_output}: {len(idx)} 行, {idx['date'].min()} ~ {idx['date'].max()}")

    # 2. 个股
    universe = load_universe(args.universe)
    rows = []
    failed = []
    t0 = time.time()
    for i, (ticker, name, sector) in enumerate(universe, 1):
        try:
            df = fetch_ticker(ticker, args.start, args.end)
            if df is None:
                failed.append(ticker)
                continue
            df.insert(1, 'ticker', ticker)
            df.insert(2, 'name', name)
            df.insert(3, 'sector', sector)
            rows.append(df)
            if i % 50 == 0 or i == len(universe):
                elapsed = time.time() - t0
                print(f"  [{i}/{len(universe)}] {ticker} ok, elapsed {elapsed:.0f}s")
        except Exception as e:
            failed.append(ticker)
            print(f"  [{i}/{len(universe)}] {ticker} 失败: {e}")
        time.sleep(args.sleep)

    if not rows:
        raise RuntimeError("没有抓到任何股票数据")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(args.output, index=False)
    print(f"\n✅ {args.output}: {len(out)} 行, {out['ticker'].nunique()} 个 ticker, "
          f"{out['sector'].nunique()} 个 sector, {out['date'].min()} ~ {out['date'].max()}")
    if failed:
        print(f"⚠️ 失败/无数据 ticker ({len(failed)}): {failed}")


if __name__ == '__main__':
    main()
