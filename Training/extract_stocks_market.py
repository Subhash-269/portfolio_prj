import pandas_datareader.data as web
from datetime import datetime
import pandas as pd

# # 30 Diversified Tickers (Tech, Finance, Health, Energy, Commodities)
# tickers = [
#     # Tech
#     "AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "INTC",
#     # Finance
#     "JPM", "BAC", "V", "MA",
#     # Healthcare
#     "JNJ", "PFE", "UNH", "MRK",
#     # Consumer
#     "AMZN", "TSLA", "KO", "PG", "WMT", "HD",
#     # Industrial & Energy
#     "XOM", "CVX", "CAT", "BA",
#     # Real Estate
#     "O", "AMT",
#     # ETFs (Market, Gold, Bonds)
#     "SPY", "QQQ", "GLD", "TLT"
# ]

survivors_df = pd.read_csv("sp500_companies.csv")
tickers = survivors_df["Symbol"].tolist()

start = datetime(2016, 1, 1)
end = datetime.now()

all_data = [] # Using a list is slightly faster for concat later

print(f"Fetching data for {len(tickers)} tickers...")

for t in tickers:
    try:
        # Fetch from Stooq
        df = web.DataReader(t, "stooq", start, end)
        
        # Clean data: Stooq returns inverted dates, so we flip it
        df = df.iloc[::-1].reset_index()
        
        # Add identifier column
        df["Ticker"] = t
        
        all_data.append(df)
        print(f"✓ Loaded: {t}")
        
    except Exception as e:
        print(f"x Failed: {t} - {e}")

if all_data:
    # Combine all into one DataFrame
    prices = pd.concat(all_data, ignore_index=True)

    # Reorder columns for readability (Date, Ticker, Close, ...)
    cols = ['Date', 'Ticker'] + [c for c in prices.columns if c not in ['Date', 'Ticker']]
    prices = prices[cols]

    print("\nSuccess! Data preview:")
    print(prices.head())
    print(f"Total Rows: {len(prices)}")

    # Save to CSV
    prices.to_csv("diversified_market_data.csv", index=False)
    print("Saved to 'diversified_market_data.csv'")
else:
    print("No data was fetched.")