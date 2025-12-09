import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_us_bond_yields(start_date, end_date):
    # --- Ticker Map ---
    # These are CBOE Interest Rate Indices
    # Note: The "Price" is actually the Yield x 10. 
    # Example: If Close is 42.00, the Yield is 4.20%
    bond_tickers = {
        "03-Month T-Bill Yield": "^IRX",
        "05-Year T-Note Yield": "^FVX",
        "10-Year T-Note Yield": "^TNX", # The Benchmark
        "30-Year T-Bond Yield": "^TYX"
    }
    
    reverse_map = {v: k for k, v in bond_tickers.items()}
    tickers_list = list(bond_tickers.values())

    print(f"Fetching Bond Yields from {start_date} to {end_date}...")

    # 1. Bulk Download
    try:
        raw_data = yf.download(
            tickers_list, 
            start=start_date, 
            end=end_date, 
            group_by='ticker', 
            auto_adjust=True, 
            progress=False
        )
    except Exception as e:
        print(f"Error during download: {e}")
        return pd.DataFrame()

    if raw_data.empty:
        return pd.DataFrame()

    # 2. Reshape (Wide to Long)
    # Stack the Ticker level into the index
    long_df = raw_data.stack(level=0)
    
    # 3. Clean Up
    long_df.reset_index(inplace=True) 
    
    # Map Ticker to Name
    long_df['Bond Type'] = long_df['Ticker'].map(reverse_map)
    
    # 4. Filter Columns
    # We keep Volume to maintain your requested format, but it will be 0 for yields.
    cols_to_keep = ['Date', 'Bond Type', 'Open', 'High', 'Low', 'Close', 'Volume']
    final_df = long_df[cols_to_keep].copy()

    # 5. Correct the Values (Optional but recommended)
    # Yahoo stores 4.2% as 42.0. Let's convert it to actual percentage (4.2)
    # Uncomment the lines below if you want the actual % value
    # numeric_cols = ['Open', 'High', 'Low', 'Close']
    # final_df[numeric_cols] = final_df[numeric_cols] / 10

    # Sort
    final_df.sort_values(by=['Bond Type', 'Date'], inplace=True)
    
    return final_df

# --- Execution ---
start = "2020-01-01"
end = datetime.today().strftime('%Y-%m-%d')

df_bonds = fetch_us_bond_yields(start, end)

if not df_bonds.empty:
    print("\n" + "="*40)
    print("       US BOND DATA REPORT       ")
    print("="*40)
    
    # Preview
    print(df_bonds.head(10))
    
    # Size Report
    print("\n--- Rows per Bond Type ---")
    print(df_bonds['Bond Type'].value_counts().sort_index())
    
    # Save
    csv_name = "us_bonds_yields.csv"
    df_bonds.to_csv(csv_name, index=False)
    print(f"\nSaved to '{csv_name}'")
    
else:
    print("No data fetched.")