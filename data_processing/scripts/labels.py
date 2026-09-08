"""Forward volatility and daily cross-sectional labels, independent of NLP models."""
import math

import numpy as np
import pandas as pd


LABEL_SCHEME = "daily_top20_forward_rv5_v1"


def compute_future_realized_vol(df_stock, window=5, ret_col="RET",
                                ticker_col="ticker", date_col="date"):
    """Use exactly the next `window` observations within each ticker.

    Missing/incomplete forward windows remain NaN, never negative labels.
    """
    if window < 1:
        raise ValueError("window must be positive")
    df = df_stock.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if df.duplicated([ticker_col, date_col]).any():
        raise ValueError("Expected one return per ticker/date")
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    returns = df.groupby(ticker_col)[ret_col]
    squares = pd.concat([returns.shift(-k).pow(2) for k in range(1, window + 1)], axis=1)
    df[f"RV_{window}"] = np.sqrt(squares.sum(axis=1, min_count=window))
    df["label_end_date"] = df.groupby(ticker_col)[date_col].shift(-window)
    return df[[ticker_col, date_col, f"RV_{window}", "label_end_date"]]


def daily_top_fraction_labels(volatility, fraction=0.2, value_col="RV_5"):
    """Rank distinct stocks per date, before joining to Reddit posts.

    Use ceil(fraction * N) positives; break equal-volatility ties by ticker.
    The universe is all stocks with complete forward returns in the input.
    """
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    df = volatility.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df.duplicated(["ticker", "date"]).any():
        raise ValueError("Rank unique ticker/date rows, not duplicated posts")
    df = df.loc[np.isfinite(df[value_col]) & df["label_end_date"].notna()].copy()
    df = df.sort_values(["date", value_col, "ticker"], ascending=[True, False, True])
    df["daily_rank"] = df.groupby("date").cumcount() + 1
    df["universe_size"] = df.groupby("date")["ticker"].transform("size")
    df["positive_count"] = df["universe_size"].map(lambda n: math.ceil(fraction * n))
    df["target"] = (df["daily_rank"] <= df["positive_count"]).astype(int)
    # A rank depends on all stocks on this date, so purge using the latest horizon.
    df["label_end_date"] = df.groupby("date")["label_end_date"].transform("max")
    return df.reset_index(drop=True)


def relabel_samples(samples, labels, boundary=None):
    """Reuse cached text windows; drop missing labels or boundary-crossing horizons."""
    indexed = labels.set_index(["ticker", "date"])
    if not indexed.index.is_unique:
        raise ValueError("Labels must be unique per ticker/date")
    mapping = indexed[["target", "label_end_date"]].to_dict("index")
    boundary = pd.Timestamp(boundary) if boundary is not None else None
    output = []
    for ticker, anchor, dates, _ in samples:
        label = mapping.get((ticker, pd.Timestamp(anchor)))
        if label is None:
            continue
        if boundary is not None and label["label_end_date"] >= boundary:
            continue
        output.append((ticker, anchor, dates, int(label["target"])))
    return output
