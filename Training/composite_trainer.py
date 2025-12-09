"""
Composite Trainer (Stocks + Commodities)

Generates three offline composite portfolios (Conservative, Balanced, Aggressive)
from US stocks (by sector) and commodities (by type). Saves results to
Training/composite_top3.json for the frontend Composite view.

Methodology overview:
- Build monthly returns per asset group (EOM resample) and daily returns for short-term stats.
- Compute per-asset daily volatility and 1W/1M momentum.
- Construct tiers:
    Conservative: inverse-volatility exposure, slight commodity de-emphasis.
    Balanced: 50% inverse-vol + 50% 1M momentum blend.
    Aggressive: momentum-driven (70% 1M, 30% 1W), commodities downweighted.
- Enforce risk ordering using monthly covariance to ensure
    vol(Cons) <= vol(Bal) <= vol(Agg).
- Metrics: Sharpe, Sortino, MaxDD, CVaR(95%), monthly/annual vol,
    short-term 1W and 1M portfolio returns.

Detailed doc with flowchart: Training/COMPOSITE_TRAINER.md
"""

import json
import os
import math
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STOCKS_CSV = BASE_DIR / 'diversified_market_data.csv'
COMMODITIES_CSV = BASE_DIR / 'Commodities' / 'commodities.csv'
MAPPER_JSON = BASE_DIR / 'mapper.json'
OUTPUT_JSON = BASE_DIR / 'composite_top3.json'

# Helper: monthly returns from daily Close using end-of-month levels per asset
def monthly_returns_from_close(df: pd.DataFrame, date_col: str, id_col: str, close_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col])
    # End-of-month close per asset
    eom = (
        df.set_index(date_col)
          .groupby(id_col)[close_col]
          .resample('ME').last()
          .rename('close')
          .reset_index()
    )
    eom['ret'] = eom.groupby(id_col)['close'].pct_change()
    eom = eom.dropna(subset=['ret'])
    eom = eom.rename(columns={date_col: 'date', id_col: 'id'})
    return eom[['date','id','ret']]

# Helper: daily returns from Close per asset
def daily_returns_from_close(df: pd.DataFrame, date_col: str, id_col: str, close_col: str) -> pd.DataFrame:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.sort_values([id_col, date_col])
    d['ret'] = d.groupby(id_col)[close_col].pct_change()
    d = d.dropna(subset=['ret'])
    d = d.rename(columns={date_col: 'date', id_col: 'id'})
    return d[['date','id','ret']]

# Stocks monthly returns aggregated to sectors using mapper.json
def stocks_monthly_returns_to_sectors(stocks_df: pd.DataFrame, mapper: dict) -> pd.DataFrame:
    mr = monthly_returns_from_close(stocks_df, 'Date', 'Ticker', 'Close')
    # Map tickers to sectors (fallback Unknown)
    def map_sector(ticker: str) -> str:
        info = mapper.get(ticker) or {}
        return info.get('sector') or info.get('Sector') or 'Unknown'
    mr['sector'] = mr['id'].apply(map_sector)
    # Sector average monthly returns
    sec = mr.groupby(['date','sector'])['ret'].mean().reset_index()
    sec = sec.rename(columns={'sector': 'asset'})
    return sec[['date','asset','ret']]

def stocks_daily_returns_to_sectors(stocks_df: pd.DataFrame, mapper: dict) -> pd.DataFrame:
    dr = daily_returns_from_close(stocks_df, 'Date', 'Ticker', 'Close')
    def map_sector(ticker: str) -> str:
        info = mapper.get(ticker) or {}
        return info.get('sector') or info.get('Sector') or 'Unknown'
    dr['sector'] = dr['id'].apply(map_sector)
    sec = dr.groupby(['date','sector'])['ret'].mean().reset_index()
    sec = sec.rename(columns={'sector': 'asset'})
    return sec[['date','asset','ret']]

# Commodities monthly returns aggregated by Commodity Type
def commodities_monthly_returns_to_types(comm_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names
    df = comm_df.rename(columns={'Commodity Type': 'CommodityType'})
    mr = monthly_returns_from_close(df, 'Date', 'CommodityType', 'Close')
    mr = mr.rename(columns={'id': 'asset'})
    return mr[['date','asset','ret']]

def commodities_daily_returns_to_types(comm_df: pd.DataFrame) -> pd.DataFrame:
    df = comm_df.rename(columns={'Commodity Type': 'CommodityType'})
    dr = daily_returns_from_close(df, 'Date', 'CommodityType', 'Close')
    dr = dr.rename(columns={'id': 'asset'})
    return dr[['date','asset','ret']]

# Combine to pivot of assets x date monthly returns
def aggregate_buckets(stocks_sectors: pd.DataFrame, commodities_types: pd.DataFrame) -> pd.DataFrame:
    all_ret = pd.concat([stocks_sectors, commodities_types], ignore_index=True)
    pivot = all_ret.pivot(index='date', columns='asset', values='ret').dropna(how='all')
    # Drop columns with too few observations
    min_obs = 12
    pivot = pivot[[c for c in pivot.columns if pivot[c].count() >= min_obs]]
    return pivot

def aggregate_buckets_daily(stocks_sectors_daily: pd.DataFrame, commodities_types_daily: pd.DataFrame) -> pd.DataFrame:
    all_ret = pd.concat([stocks_sectors_daily, commodities_types_daily], ignore_index=True)
    pivot = all_ret.pivot(index='date', columns='asset', values='ret').dropna(how='all')
    # Filter assets with enough daily observations
    min_obs = 60
    pivot = pivot[[c for c in pivot.columns if pivot[c].count() >= min_obs]]
    return pivot

def metrics(returns: pd.DataFrame, weights: pd.Series, daily_returns: pd.DataFrame | None = None):
    # Monthly metrics
    portfolio_ret = returns.fillna(0).dot(weights)
    mean = portfolio_ret.mean()
    vol = portfolio_ret.std(ddof=1)
    sharpe = mean / vol if vol and vol > 0 else 0.0
    downside = portfolio_ret[portfolio_ret < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = mean / downside_std if downside_std and downside_std > 0 else 0.0
    # Max Drawdown
    cum = (1 + portfolio_ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1).min() if len(cum) else 0.0
    # CVaR 95%
    q = portfolio_ret.quantile(0.05) if len(portfolio_ret) else 0.0
    cvar = portfolio_ret[portfolio_ret <= q].mean() if len(portfolio_ret) else 0.0
    # Annualized measures
    vol_annual = float(vol) * (12 ** 0.5) if vol and vol > 0 else 0.0

    # Short-term returns (portfolio daily if provided)
    short_1w = None
    short_1m = None
    if daily_returns is not None and set(weights.index).issubset(set(daily_returns.columns)):
        daily_port = daily_returns.fillna(0).dot(weights)
        if len(daily_port) >= 5:
            short_1w = float((daily_port.tail(5) + 1.0).prod() - 1.0)
        if len(daily_port) >= 21:
            short_1m = float((daily_port.tail(21) + 1.0).prod() - 1.0)

    result = {
        'mean_monthly': float(mean),
        'vol_monthly': float(vol),
        'vol_annual': float(vol_annual),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(dd),
        'cvar_95': float(cvar)
    }
    if short_1w is not None:
        result['short_term_1w'] = short_1w
    if short_1m is not None:
        result['short_term_1m'] = short_1m
    return result

def normalize(weights: pd.Series, cap=0.4, min_alloc=0.0):
    w = weights.clip(lower=min_alloc)
    if cap is not None:
        w = w.clip(upper=cap)
    s = w.sum()
    if s <= 0:
        # Equal weight fallback
        w = pd.Series(1.0, index=w.index)
        s = w.sum()
    return w / s

def construct_tiers(returns: pd.DataFrame, daily_returns: pd.DataFrame | None = None):
    assets = list(returns.columns)
    # Classify commodities by name heuristics
    def is_comm_name(a: str) -> bool:
        al = a.lower()
        return any(k in al for k in ['commodity','oil','gold','gas','wheat','corn','copper','brent','heating'])
    is_commodity = [is_comm_name(a) for a in assets]

    # Compute per-asset daily vol and momentum if available
    vol_asset = pd.Series(0.0, index=assets)
    mom_1w = pd.Series(0.0, index=assets)
    mom_1m = pd.Series(0.0, index=assets)
    if daily_returns is not None:
        for a in assets:
            s = daily_returns[a].dropna() if a in daily_returns.columns else pd.Series(dtype=float)
            if len(s) >= 21:
                vol_asset[a] = float(s.std(ddof=1))
                mom_1w[a] = float((s.tail(5) + 1.0).prod() - 1.0)
                mom_1m[a] = float((s.tail(21) + 1.0).prod() - 1.0)

    # Precompute monthly covariance for risk calculations
    cov_m = returns.cov().fillna(0.0)

    def port_vol_annual(w: pd.Series) -> float:
        wv = w.reindex(assets).fillna(0.0).values
        cov = cov_m.reindex(index=assets, columns=assets).values
        vol_m = float((wv @ cov @ wv) ** 0.5) if cov.size else 0.0
        return vol_m * (12 ** 0.5)

    # Conservative: inverse volatility weighting, slightly de-emphasize commodities to lower risk
    inv_vol = vol_asset.replace(0, pd.NA)
    inv_vol = 1.0 / inv_vol
    inv_vol = inv_vol.fillna(0.0)
    cons_raw = inv_vol.copy()
    # Reduce commodity share slightly to avoid elevated short-term volatility
    cons_raw[[a for a, ic in zip(assets, is_commodity) if ic]] *= 0.9
    cons = normalize(cons_raw, cap=0.5, min_alloc=0.0)

    # Balanced: blend inverse-vol (50%) + momentum-1m (50%)
    mom_pos = mom_1m.clip(lower=0.0)
    bal_raw = 0.5 * (cons_raw / (cons_raw.sum() or 1)) + 0.5 * (mom_pos / (mom_pos.sum() or 1))
    bal = normalize(bal_raw, cap=0.5, min_alloc=0.0)

    # Aggressive: momentum driven (1m primary, 1w secondary), lower commodities share
    mom_combo = 0.7 * mom_1m + 0.3 * mom_1w
    mom_combo = mom_combo.clip(lower=0.0)
    agg_raw = mom_combo.copy()
    # Downweight commodities to ensure higher risk/return from equities
    agg_raw[[a for a, ic in zip(assets, is_commodity) if ic]] *= 0.6
    # If momentum all zeros, fallback to equal-weight stocks heavy
    if agg_raw.sum() <= 0:
        agg_raw = pd.Series({a: (0.1 if ic else 0.9 / max(len(assets) - sum(is_commodity), 1)) for a, ic in zip(assets, is_commodity)})
    agg = normalize(agg_raw, cap=0.4, min_alloc=0.0)

    # Enforce intuitive risk ordering: vol(cons) <= vol(bal) <= vol(agg)
    v_cons = port_vol_annual(cons)
    v_bal = port_vol_annual(bal)
    v_agg = port_vol_annual(agg)

    # Adjust balanced by mixing between cons and agg if needed
    if not (v_cons <= v_bal <= v_agg):
        # target roughly mid risk between cons and agg
        target = v_cons + 0.5 * max(0.0, (v_agg - v_cons))
        # search t in [0,1] for w = (1-t)*cons + t*agg
        t_best = 0.5
        w_best = normalize((1 - t_best) * cons + t_best * agg)
        v_best = port_vol_annual(w_best)
        for t in [i/10 for i in range(0, 11)]:
            w_t = normalize((1 - t) * cons + t * agg)
            v_t = port_vol_annual(w_t)
            if abs(v_t - target) < abs(v_best - target):
                t_best, w_best, v_best = t, w_t, v_t
        bal = w_best
        v_bal = v_best

    # If aggressive is not the highest risk, nudge it toward momentum
    if v_agg < max(v_bal, v_cons):
        # amplify momentum tilt
        agg_boost = agg_raw.copy()
        agg_boost[[a for a, ic in zip(assets, is_commodity) if ic]] *= 0.5
        agg = normalize(agg_boost, cap=0.45, min_alloc=0.0)

    tiers = []
    for name, w in [('Conservative', cons), ('Balanced', bal), ('Aggressive', agg)]:
        m = metrics(returns, w, daily_returns=daily_returns)
        tiers.append({
            'tier': name,
            'weights': {k: float(v) for k, v in w.items()},
            'metrics': m
        })
    return tiers

def main():
    # Load data
    if not STOCKS_CSV.exists() or not COMMODITIES_CSV.exists():
        raise FileNotFoundError('Required data files not found')
    stocks_df = pd.read_csv(STOCKS_CSV)
    commodities_df = pd.read_csv(COMMODITIES_CSV)

    # Load mapper for sector mapping
    mapper = {}
    if MAPPER_JSON.exists():
        try:
            with open(MAPPER_JSON, 'r', encoding='utf-8') as f:
                mapper = json.load(f)
        except Exception:
            mapper = {}

    stocks_sectors = stocks_monthly_returns_to_sectors(stocks_df, mapper)
    commodities_types = commodities_monthly_returns_to_types(commodities_df)
    pivot = aggregate_buckets(stocks_sectors, commodities_types)

    # Build daily pivot for short-term metrics
    stocks_sectors_daily = stocks_daily_returns_to_sectors(stocks_df, mapper)
    commodities_types_daily = commodities_daily_returns_to_types(commodities_df)
    pivot_daily = aggregate_buckets_daily(stocks_sectors_daily, commodities_types_daily)

    # Align daily to monthly assets intersection
    common_assets = [a for a in pivot.columns if a in pivot_daily.columns]
    pivot_daily = pivot_daily[common_assets]
    pivot = pivot[common_assets]

    tiers = construct_tiers(pivot, daily_returns=pivot_daily)

    # Save
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': pd.Timestamp.now().isoformat(),
            'assets': list(pivot.columns),
            'top3': tiers,
            'notes': 'Heuristic composite trainer (Stocks + Commodities); replace with optimization later.'
        }, f, indent=2)
    print(f'Wrote {OUTPUT_JSON}')

if __name__ == '__main__':
    main()
