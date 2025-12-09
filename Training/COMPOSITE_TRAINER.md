# Composite Trainer Methodology (Stocks + Commodities)

Purpose: Generate three offline composite portfolios (Conservative, Balanced, Aggressive) and save them to `Training/composite_top3.json` for display in the Composite section (no API calls required).

## Inputs
- Stocks: `Training/diversified_market_data.csv` with `Date, Ticker, Close` (other OHLC columns may exist).
- Commodities: `Training/Commodities/commodities.csv` with `Date, Commodity Type, Close`.
- Sector mapping: `Training/mapper.json` mapping `Ticker -> sector`.

## Outputs
- `Training/composite_top3.json` containing:
  - `assets`: list of asset groups (stock sectors + commodity types)
  - `top3`: array of 3 entries `{ tier, weights, metrics }`
  - `metrics`: `mean_monthly, vol_monthly, vol_annual, sharpe, sortino, max_drawdown, cvar_95, short_term_1w, short_term_1m`

## High-Level Steps
1. Load CSVs and `mapper.json`.
2. Compute monthly returns per asset group:
   - End-of-month (ME) resample of `Close` per asset; `ret = pct_change`.
   - Stocks aggregated by sector; commodities by `Commodity Type`.
3. Compute daily returns per asset group for short-term metrics:
   - Daily `pct_change` on `Close`, grouped as above.
4. Align assets present in both monthly and daily pivots.
5. Compute per-asset daily stats:
   - Volatility `σ_d(asset)` and momentum (`1W`, `1M`).
6. Construct candidate weights for each tier:
   - Conservative: inverse-volatility with slight commodity de-emphasis; caps; normalize.
   - Balanced: 50% inverse-vol + 50% 1M momentum; caps; normalize.
   - Aggressive: momentum-driven (70% 1M + 30% 1W), downweight commodities; caps; normalize.
7. Enforce risk ordering using monthly covariance `Cov_m`:
   - Annualized risk `σ_annual(w) = sqrt(w^T Cov_m w) * sqrt(12)`.
   - If `σ_cons <= σ_bal <= σ_agg` is violated, mix `Balanced` between `Conservative` and `Aggressive` to target mid risk; nudge `Aggressive` higher if needed.
8. Compute portfolio-level metrics for each tier from monthly and daily series.
9. Save the JSON.

## Flowchart (Mermaid)
```mermaid
flowchart TD
    A[Start] --> B[Load Stocks, Commodities, mapper.json]
    B --> C[Monthly EOM returns per asset group]
    B --> D[Daily returns per asset group]
    C --> E[Align assets; build pivots R_m & R_d]
    D --> E
    E --> F[Per-asset daily stats: vol & 1W/1M momentum]
    F --> G[Build Conservative weights (inverse-vol; slight commodity downweight)]
    F --> H[Build Balanced weights (50% inv-vol + 50% 1M mom)]
    F --> I[Build Aggressive weights (0.7*1M + 0.3*1W; commodity downweight)]
    G --> J[Risk ordering via Cov(R_m)]
    H --> J
    I --> J
    J --> K[Compute metrics (Sharpe, Sortino, MaxDD, CVaR, vol, 1W/1M)]
    K --> L[Write Training/composite_top3.json]
    L --> M[End]
```

## Metrics
- mean_monthly: average monthly return.
- vol_monthly: monthly standard deviation.
- vol_annual: annualized volatility `vol_monthly * sqrt(12)`.
- Sharpe: `mean_monthly / vol_monthly`.
- Sortino: `mean_monthly / downside_std`.
- Max Drawdown: peak-to-trough drop over monthly series.
- CVaR(95%): average of worst 5% monthly returns.
- short_term_1w: ∏(1+daily_r) over last 5 trading days − 1.
- short_term_1m: ∏(1+daily_r) over last ~21 trading days − 1.

## Tuning Knobs
- Per-asset caps and minimum allocations.
- Commodity tilt strength per tier.
- Momentum windows and blend ratios.
- Risk-ordering tolerance or explicit target risk buckets (e.g., 10%, 15%, 20% vol).

## Usage
Run the trainer and refresh the frontend data:

```powershell
python "c:\Users\subha\Desktop\Workspace\Personal\PortfolioOPtProject\Training\composite_trainer.py"
copy "c:\Users\subha\Desktop\Workspace\Personal\PortfolioOPtProject\Training\composite_top3.json" "c:\Users\subha\Desktop\Workspace\Personal\PortfolioOPtProject\front_end\public\composite_top3.json"
```

Open the app and switch to the Composite tab to view the portfolios.
