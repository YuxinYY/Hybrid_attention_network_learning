"""
构建 sector 级异质波动率（idiosyncratic volatility, IVOL）标签。

⚠️ 历史流程（FF3 三因子 margin 标签）:
    当前主流程已改为 pipeline.py 内置的 CAPM 中位数二分类标签（capm，
    预测未来 5 日异质波动率高低），train/val/test_samples.pt 直接由
    pipeline.py 输出。本脚本仅保留旧的三类 FF3 margin 标签（ts_rise /
    xs_median / xs_demeaned）作为历史参考。

为什么用异质波动率:
    行业组合的总波动率里绝大部分是市场/规模/价值等共同因子驱动的。直接预测行业总波动率，
    模型很容易只学到"市场整体处于高波动 regime"，而这跟 Reddit 上讨论的是哪个行业无关。
    剥离因子后剩下的残差波动，才是"这个行业自己出了事"的部分——也正是社交媒体讨论
    可能提供增量信息的地方。

定义（Ang, Hodrick, Xing & Zhang 2006 的 sector 组合版）:
    1. sector 等权日收益 R_{s,t}：该 sector 下全部 ticker 当日 RET 的等权平均。
    2. 因子暴露用 **trailing 窗口** 估计（截至 t，不含未来）:
           R_{s,u} - RF_u = a_s + b_s*MKT_u + h1_s*SMB_u + h2_s*HML_u + e_{s,u},
           u in [t-BETA_WINDOW+1, t]
       AHXZ 原文是在同一个月内做回归再取残差 std；这里改成 trailing beta，原因是
       用仅 20 个观测做回归时，真正的异质冲击会被部分吸收进 beta 估计，反而把我们
       想要的信号衰减掉。固定暴露后，残差才纯粹是"未被因子解释的部分"。
    3. 目标 = 未来 H 天残差的标准差（年化）:
           IVOL_fwd(s,t) = std( e_{s,t+1..t+H} ) * sqrt(252)
       其中残差用 t 时刻已知的 (a_s, b_s, h1_s, h2_s) 计算。
    4. 同样的暴露，回看 BASE_WINDOW 天算出当前水平（全部落在过去）:
           IVOL_past(s,t) = std( e_{s,t-BASE_WINDOW+1..t} ) * sqrt(252)

二分类 margin（下游统一用 threshold=0.0，>0 记为 1）:
    - ts_rise   = IVOL_fwd - IVOL_past
        含义: 该行业未来一个月的异质波动会不会高于它当前的水平（时序 regime 变化）。
    - xs_median = IVOL_fwd - median_over_sectors(IVOL_fwd at same date)
        含义: 该行业未来一个月的异质波动会不会高于同日其他行业（截面排序）。
        每天恰好一半为 1，天然剔除了全市场共同波动 regime，是对"文本能否区分行业"
        更严格的检验。

用法:
    python data_processing/scripts/build_ivol_labels.py \
        --stocks data_processing/stocks_2020_2023.csv \
        --output data_processing/sector_ivol_labels.csv

输出列: sector, date, ivol_fwd, ivol_past, ts_rise, xs_median
"""

import argparse
import io
import os
import zipfile

import numpy as np
import pandas as pd
import urllib.request

FF3_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_daily_CSV.zip")
FACTORS = ["Mkt-RF", "SMB", "HML"]


def load_ff3(cache_path: str) -> pd.DataFrame:
    """下载/读取 Ken French 日频三因子，返回 date, Mkt-RF, SMB, HML, RF（小数，非百分数）。"""
    if not os.path.exists(cache_path):
        print(f"下载 Fama-French 日频三因子 -> {cache_path}")
        with urllib.request.urlopen(FF3_URL) as resp:
            blob = resp.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            name = zf.namelist()[0]
            raw = zf.read(name).decode("utf-8", errors="ignore")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(raw)
    else:
        print(f"复用本地因子缓存: {cache_path}")
        raw = open(cache_path, encoding="utf-8").read()

    # 文件开头是若干行说明文字，末尾还附了年频数据；只取形如 YYYYMMDD 开头的日频行
    rows = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 5 and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)

    df = pd.DataFrame(rows, columns=["date"] + FACTORS + ["RF"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date
    for c in FACTORS + ["RF"]:
        df[c] = pd.to_numeric(df[c]) / 100.0  # 原文件单位是百分数
    print(f"因子数据: {len(df)} 行, {df['date'].min()} ~ {df['date'].max()}")
    return df


def build_sector_returns(stocks_path: str) -> pd.DataFrame:
    """sector 等权日收益。返回 sector, date, sector_RET。"""
    df = pd.read_csv(stocks_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
    df = df.dropna(subset=["RET", "sector"])
    df = df[df["sector"] != "MISSING"]

    out = (df.groupby(["sector", "date"])["RET"]
             .mean()
             .reset_index()
             .rename(columns={"RET": "sector_RET"}))
    n_names = df.groupby(["sector", "date"])["ticker"].nunique().reset_index(name="n_names")
    out = out.merge(n_names, on=["sector", "date"])
    print(f"sector 等权收益: {len(out)} 行, {out['sector'].nunique()} 个 sector, "
          f"{out['date'].min()} ~ {out['date'].max()}")
    return out


def compute_ivol(panel: pd.DataFrame, horizon: int, beta_window: int,
                 beta_min: int, base_window: int) -> pd.DataFrame:
    """
    对每个 sector 逐日估计 trailing 因子暴露，并算出未来/过去残差波动。
    panel 需含: sector, date, sector_RET, Mkt-RF, SMB, HML, RF
    """
    results = []
    ann = np.sqrt(252.0)

    for sector, g in panel.groupby("sector", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        exret = (g["sector_RET"] - g["RF"]).to_numpy(dtype=float)
        F = g[FACTORS].to_numpy(dtype=float)
        # 设计矩阵加截距列
        X = np.column_stack([np.ones(len(g)), F])
        dates = g["date"].to_numpy()
        n = len(g)

        rows = []
        for i in range(n):
            # --- trailing 窗口估计暴露：只用 [i-beta_window+1, i] ---
            lo = max(0, i - beta_window + 1)
            Xw, yw = X[lo:i + 1], exret[lo:i + 1]
            if len(yw) < beta_min or not np.isfinite(yw).all():
                continue
            coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

            # --- 未来 H 天残差（暴露固定为 t 时刻已知值）---
            fh, fe = i + 1, i + 1 + horizon
            if fe > n:
                continue
            resid_fwd = exret[fh:fe] - X[fh:fe] @ coef
            if not np.isfinite(resid_fwd).all():
                continue
            ivol_fwd = float(np.std(resid_fwd, ddof=1) * ann)

            # --- 过去 base_window 天残差（同一组暴露，全部落在过去）---
            bl = i - base_window + 1
            if bl < 0:
                continue
            resid_past = exret[bl:i + 1] - X[bl:i + 1] @ coef
            ivol_past = float(np.std(resid_past, ddof=1) * ann)

            rows.append((sector, dates[i], ivol_fwd, ivol_past))

        print(f"  {sector:<24} 可用标签 {len(rows)} 天")
        results.extend(rows)

    out = pd.DataFrame(results, columns=["sector", "date", "ivol_fwd", "ivol_past"])
    return out


def main():
    p = argparse.ArgumentParser(description="构建 sector 级异质波动率标签")
    p.add_argument("--stocks", default="data_processing/stocks_2020_2023.csv",
                   help="含 date,ticker,sector,RET 的股票面板（需比样本期多一年历史用于估 beta）")
    p.add_argument("--output", default="data_processing/sector_ivol_labels.csv")
    p.add_argument("--ff3_cache", default="data_processing/ff3_daily.csv")
    p.add_argument("--horizon", type=int, default=20, help="预测窗口（交易日）")
    p.add_argument("--beta_window", type=int, default=250, help="估计因子暴露的 trailing 窗口")
    p.add_argument("--beta_min", type=int, default=120, help="估 beta 所需最少观测数")
    p.add_argument("--base_window", type=int, default=60,
                   help="ts_rise 基准：回看多少天算当前异质波动水平")
    args = p.parse_args()

    ff3 = load_ff3(args.ff3_cache)
    sec = build_sector_returns(args.stocks)

    panel = sec.merge(ff3, on="date", how="inner")
    print(f"对齐因子后: {len(panel)} 行")

    print(f"\n估计 trailing 暴露 (window={args.beta_window}, min={args.beta_min}) "
          f"并计算 IVOL (H={args.horizon}, base={args.base_window}) ...")
    ivol = compute_ivol(panel, args.horizon, args.beta_window, args.beta_min, args.base_window)

    # --- 三种二分类 margin ---
    ivol["ts_rise"] = ivol["ivol_fwd"] - ivol["ivol_past"]
    ivol["xs_median"] = ivol["ivol_fwd"] - ivol.groupby("date")["ivol_fwd"].transform("median")

    # xs_demeaned: 先把每个行业的波动换算成"相对它自己当前水平的变化倍数"（取对数使其对称），
    # 再和同日其他行业比。这样既剔除了全市场波动 regime（截面比较），也剔除了行业固有的
    # 波动水平差异（除以自身 ivol_past）。
    # 加这一条的原因: xs_median 几乎完全由"这是哪个行业"决定——Real Estate 的异质波动
    # 长期就是 Consumer Cyclical 的两倍多，模型只要从文本认出行业就能拿高分，
    # 高准确率并不代表预测到了任何东西。
    log_ratio = np.log(ivol["ivol_fwd"].clip(lower=1e-8) / ivol["ivol_past"].clip(lower=1e-8))
    ivol["log_ratio"] = log_ratio
    ivol["xs_demeaned"] = log_ratio - ivol.groupby("date")["log_ratio"].transform("median")

    ivol = ivol.sort_values(["sector", "date"]).reset_index(drop=True)
    ivol.to_csv(args.output, index=False)

    print(f"\n✅ 已写出 {args.output}: {len(ivol)} 行, "
          f"{ivol['date'].min()} ~ {ivol['date'].max()}")
    print("\nIVOL 分布 (年化):")
    print(ivol.groupby("sector")[["ivol_fwd", "ivol_past"]].mean().round(4))
    print("\n正类占比:")
    for col in ["ts_rise", "xs_median", "xs_demeaned"]:
        pos = (ivol[col] > 0).mean()
        print(f"  {col:<10} 正类 {pos:.1%}")


if __name__ == "__main__":
    main()
