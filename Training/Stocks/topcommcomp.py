import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_enriched_sp500_history(years=5):
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"1. Fetching S&P 500 base data from Wikipedia...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        html_data = StringIO(response.text)
        tables = pd.read_html(html_data)
        
        # ==========================================
        # PART 1: The Current List (Detailed)
        # ==========================================
        current_df = tables[0]
        
        # Target Columns
        target_cols = [
            'Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 
            'Headquarters Location', 'Date added', 'CIK', 'Founded'
        ]
        
        # Create Master List
        master_list = current_df[target_cols].copy()
        master_list['Status'] = 'Active'
        
        print(f"   -> Found {len(master_list)} currently active companies.")
        
        # ==========================================
        # PART 2: The Removed List
        # ==========================================
        changes_df = tables[1].copy()
        
        # Flatten Headers
        new_columns = []
        for col in changes_df.columns:
            if isinstance(col, tuple):
                new_columns.append(f"{col[0]}_{col[1]}".strip())
            else:
                new_columns.append(col)
        changes_df.columns = new_columns
        
        # Identify Columns dynamically
        date_col = next(c for c in changes_df.columns if "Date" in c)
        removed_ticker_col = next(c for c in changes_df.columns if "Removed" in c and "Ticker" in c)
        removed_name_col = next(c for c in changes_df.columns if "Removed" in c and "Security" in c)
        
        # Filter Dates
        changes_df[date_col] = pd.to_datetime(changes_df[date_col], errors='coerce')
        cutoff_date = datetime.now() - timedelta(days=years*365)
        
        # Get Removed Companies
        recent_removals = changes_df[changes_df[date_col] >= cutoff_date].copy()
        removed_companies = recent_removals[[removed_ticker_col, removed_name_col]].dropna()
        removed_companies.columns = ['Symbol', 'Security']
        
        # Initialize missing columns with placeholders
        for col in target_cols:
            if col not in removed_companies.columns:
                removed_companies[col] = None 
        
        removed_companies['Status'] = 'Removed'
        
        print(f"2. Found {len(removed_companies)} removed companies.")
        print(f"3. Attempting to fetch missing Sector/HQ data via Yahoo Finance...")
        
        # ==========================================
        # PART 3: ENRICH DATA WITH YFINANCE
        # ==========================================
        # We process tickers in batches to populate Sector/Industry/HQ
        
        # Create a list of tickers to fetch
        tickers_list = removed_companies['Symbol'].tolist()
        
        # Fetch data in bulk (much faster than looping)
        try:
            tickers_obj = yf.Tickers(' '.join(tickers_list))
            
            for idx, row in removed_companies.iterrows():
                ticker = row['Symbol']
                try:
                    # Access the info dictionary
                    info = tickers_obj.tickers[ticker].info
                    
                    # Fill GICS Sector
                    if 'sector' in info:
                        removed_companies.at[idx, 'GICS Sector'] = info['sector']
                    else:
                        removed_companies.at[idx, 'GICS Sector'] = "Delisted/Private"

                    # Fill Sub-Industry
                    if 'industry' in info:
                        removed_companies.at[idx, 'GICS Sub-Industry'] = info['industry']
                        
                    # Fill Headquarters
                    city = info.get('city', '')
                    state = info.get('state', '')
                    country = info.get('country', '')
                    if city and state:
                        removed_companies.at[idx, 'Headquarters Location'] = f"{city}, {state}, {country}"
                    
                    # Try to fill CIK if available
                    if 'cik' in info:
                         removed_companies.at[idx, 'CIK'] = info['cik']

                except Exception:
                    # If fetch fails (completely dead ticker), leave as Delisted
                    removed_companies.at[idx, 'GICS Sector'] = "Delisted/Private"
                    
        except Exception as e:
            print(f"   Warning: YFinance fetch failed ({e}). Proceeding with partial data.")

        # ==========================================
        # PART 4: MERGE
        # ==========================================
        combined_df = pd.concat([master_list, removed_companies], ignore_index=True)
        
        # Deduplicate: Keep Active if duplicates exist
        combined_df.sort_values(by='Status', inplace=True) 
        combined_df = combined_df.drop_duplicates(subset=['Symbol'], keep='first')
        
        # Final Sort
        combined_df.sort_values(by=['Status', 'Symbol'], inplace=True)
        
        return combined_df

    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df_all = get_enriched_sp500_history(years=5)
    
    if df_all is not None:
        print("\n" + "="*40)
        print(f"      FINAL ENRICHED SUMMARY")
        print("="*40)
        print(f"Total Unique Companies: {len(df_all)}")
        
        print("\n--- Status Breakdown ---")
        print(df_all['Status'].value_counts())
        
        print("\n--- Preview of REMOVED companies (with fetched data) ---")
        # Show rows where we successfully found a sector
        removed_with_data = df_all[
            (df_all['Status'] == 'Removed') & 
            (df_all['GICS Sector'] != "Delisted/Private")
        ]
        
        if not removed_with_data.empty:
            print(removed_with_data[['Symbol', 'Security', 'GICS Sector', 'Headquarters Location']].head(10).to_string(index=False))
        else:
            print("Could not fetch extra data (Are these tickers all delisted?)")
            
        print("\n--- Preview of DELISTED companies (Data unavailable) ---")
        delisted = df_all[df_all['GICS Sector'] == "Delisted/Private"]
        print(delisted[['Symbol', 'Security', 'GICS Sector']].head(5).to_string(index=False))

        # Save to CSV
        filename = "sp500_enriched_history_5yr.csv"
        df_all.to_csv(filename, index=False)
        print(f"\nSaved enriched list to '{filename}'")