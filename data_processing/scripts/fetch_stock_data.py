"""
用 yfinance 抓取 WSB 热门股票的日线数据，产出 pipeline 所需的 stocks.csv。

输出列: date, ticker, name, RET, VOL
- RET: 日收盘收益率 (adj close pct_change)
- VOL: 成交量

用法:
    python data_processing/scripts/fetch_stock_data.py \
        --start 2022-12-01 --end 2024-01-15 --output stocks.csv
"""
import argparse
import pandas as pd
import yfinance as yf

# WSB 热门 ticker 及公司名（company name 供 GLiNER 实体匹配）
TICKERS = {
    'GME': 'GameStop',
    'AMC': 'AMC Entertainment',
    'TSLA': 'Tesla',
    'AAPL': 'Apple',
    'NVDA': 'NVIDIA',
    'PLTR': 'Palantir',
    'MSFT': 'Microsoft',
    'AMZN': 'Amazon',
    'AMD': 'Advanced Micro Devices',
    'SPY': 'S&P 500 ETF',
    'BB': 'BlackBerry',
    'NOK': 'Nokia',
    'SOFI': 'SoFi Technologies',
    'HOOD': 'Robinhood',
    'COIN': 'Coinbase',
    'MARA': 'Marathon Digital',
    'RIVN': 'Rivian',
    'LCID': 'Lucid Motors',
    'F': 'Ford',
    'INTC': 'Intel',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-12-01')
    parser.add_argument('--end', default='2024-01-15')
    parser.add_argument('--output', default='stocks.csv')
    args = parser.parse_args()

    rows = []
    for ticker, name in TICKERS.items():
        print(f"Fetching {ticker} ...")
        hist = yf.Ticker(ticker).history(start=args.start, end=args.end, auto_adjust=True)
        if hist.empty:
            print(f"  ⚠️ {ticker} 无数据，跳过")
            continue
        hist = hist.reset_index()
        hist['RET'] = hist['Close'].pct_change()
        hist = hist.dropna(subset=['RET'])
        for _, r in hist.iterrows():
            rows.append({
                'date': r['Date'].strftime('%Y-%m-%d'),
                'ticker': ticker,
                'name': name,
                'RET': r['RET'],
                'VOL': r['Volume'],
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"✅ 已写出 {args.output}: {len(df)} 行, {df['ticker'].nunique()} 个 ticker, "
          f"日期范围 {df['date'].min()} ~ {df['date'].max()}")


if __name__ == '__main__':
    main()
