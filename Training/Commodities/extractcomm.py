import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_long_format_commodities(start_date, end_date):
    # 1. Define the Ticker Map (Commodity Name -> Ticker Symbol)
    ticker_map = {
        "Crude Oil (WTI)": "CL=F",
        "Brent Crude": "BZ=F",
        "Natural Gas": "NG=F",
        "Gasoline (RBOB)": "RB=F",
        "Heating Oil": "HO=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Copper": "HG=F",
        "Soybeans": "ZS=F",
        "Wheat": "ZW=F",
        "Coal (Proxy)": "ARCH",  # Proxy: Arch Resources
        "Steel (Proxy)": "SLX",   # Proxy: VanEck Steel ETF
        "TTF Gas (Dutch)": "TFM=F",
        "Lumber (Proxy)": "WOOD", # Proxy: Global Timber ETF
        "Iron Ore (Proxy)": "RIO",# Proxy: Rio Tinto
        # "USD/CNY": "CNY=X"
    }
    
    # Reverse map for later renaming (Ticker -> Name)
    reverse_map = {v: k for k, v in ticker_map.items()}
    tickers_list = list(ticker_map.values())

    print(f"Fetching data for {len(tickers_list)} commodities...")

    # 2. Bulk Download (Group by Ticker to ensure structure)
    # group_by='ticker' makes columns: (Ticker, PriceType) e.g., ('CL=F', 'Close')
    raw_data = yf.download(
        tickers_list, 
        start=start_date, 
        end=end_date, 
        group_by='ticker', 
        auto_adjust=True, 
        progress=False
    )

    # 3. Reshape from Wide to Long
    # Stack level 0 (the Tickers) into the index
    # New Index: (Date, Ticker)
    long_df = raw_data.stack(level=0)
    
    # 4. Clean Up
    long_df.reset_index(inplace=True) # Convert Index to Columns
    
    # Map the cryptic Ticker (CL=F) back to readable Name (Crude Oil)
    long_df['Commodity Type'] = long_df['Ticker'].map(reverse_map)
    
    # 5. Format Columns as requested
    # Note: 'Close' is already adjusted because we used auto_adjust=True
    cols_to_keep = ['Date', 'Commodity Type', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Handle cases where some columns might be missing (rare, but good for safety)
    final_df = long_df[cols_to_keep].copy()

    # Sort for readability
    final_df.sort_values(by=['Commodity Type', 'Date'], inplace=True)
    
    return final_df

# --- Execution ---
start = "2016-01-01"
end = datetime.today().strftime('%Y-%m-%d')

df = fetch_long_format_commodities(start, end)

if not df.empty:
    print("\n--- Data Preview (Long Format) ---")
    print(df.head(10))
    
    # Check Volume Data
    print("\n--- Volume Data Check ---")
    # Show average volume per commodity to verify we have data
    print(df.groupby('Commodity Type')['Volume'].mean().astype(int))

    df.to_csv("commodities.csv", index=False)
    print("\nSaved to 'commodities.csv'")
else:
    print("No data found. Please check your connection.")

print("\n" + "="*40)
print("       DATASET SIZE REPORT       ")
print("="*40)

# 1. Total Size
print(f"Total Rows in Dataset: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print("-" * 40)

# 2. Size per Commodity
# This counts how many days of data we successfully fetched for each item
counts = df['Commodity Type'].value_counts().sort_index()
print("Rows per Commodity:")
print(counts)

print("-" * 40)

# 3. Check for Data Imbalance
# Find the max and min to see if any commodity is severely lagging
max_count = counts.max()
min_count = counts.min()
diff = max_count - min_count

if diff > 0:
    print(f"Note: There is a difference of {diff} days between the most and least populated commodities.")
    print("Reason: Different markets (Stock vs Futures) have different holidays.")
else:
    print("Perfect Balance: All commodities have the exact same number of days.")