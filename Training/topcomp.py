import pandas as pd
import requests
from bs4 import BeautifulSoup

def fetch_sp500_wikipedia():
    """
    Fetch S&P 500 companies from Wikipedia with proper headers.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse with pandas
        tables = pd.read_html(response.text)
        sp500_df = tables[0]
        
        print(f"Successfully fetched {len(sp500_df)} S&P 500 companies")
        print(f"\nColumns: {', '.join(sp500_df.columns)}")
        
        return sp500_df
    
    except Exception as e:
        print(f"Wikipedia method failed: {e}")
        return None

def fetch_sp500_from_url():
    """
    Alternative method using a GitHub-hosted list of S&P 500 tickers.
    """
    url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    
    try:
        df = pd.read_csv(url)
        print(f"Successfully fetched {len(df)} S&P 500 companies from GitHub")
        print(f"\nColumns: {', '.join(df.columns)}")
        return df
    
    except Exception as e:
        print(f"GitHub method failed: {e}")
        return None

def fetch_sp500_beautifulsoup():
    """
    Fetch using BeautifulSoup as a backup method.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        
        if table is None:
            table = soup.find('table', {'class': 'wikitable'})
        
        df = pd.read_html(str(table))[0]
        
        print(f"Successfully fetched {len(df)} S&P 500 companies using BeautifulSoup")
        return df
    
    except Exception as e:
        print(f"BeautifulSoup method failed: {e}")
        return None

def display_sample_companies(df, n=10):
    """Display a sample of companies from the dataframe."""
    if df is not None and len(df) > 0:
        print(f"\n--- First {n} Companies ---")
        print(df.head(n).to_string(index=False))

def save_to_csv(df, filename='sp500_companies.csv'):
    """Save the dataframe to a CSV file."""
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")

def get_companies_by_sector(df, sector):
    """Filter companies by sector."""
    if df is not None:
        # Handle different column names
        sector_col = 'GICS Sector' if 'GICS Sector' in df.columns else 'Sector'
        
        if sector_col in df.columns:
            sector_companies = df[df[sector_col] == sector]
            print(f"\n--- Companies in {sector} Sector ---")
            print(f"Total: {len(sector_companies)} companies")
            return sector_companies
    return None

# Main execution
if __name__ == "__main__":
    sp500_data = None
    
    # Try multiple methods in order
    print("Attempting to fetch S&P 500 data...\n")
    
    # Method 1: Wikipedia with pandas
    print("Method 1: Wikipedia with pandas...")
    sp500_data = fetch_sp500_wikipedia()
    
    # Method 2: GitHub CSV
    if sp500_data is None:
        print("\nMethod 2: GitHub repository...")
        sp500_data = fetch_sp500_from_url()
    
    # Method 3: BeautifulSoup
    if sp500_data is None:
        print("\nMethod 3: BeautifulSoup parsing...")
        sp500_data = fetch_sp500_beautifulsoup()
    
    if sp500_data is not None:
        # Display sample companies
        display_sample_companies(sp500_data)
        
        # Show sectors if available
        sector_col = 'GICS Sector' if 'GICS Sector' in sp500_data.columns else 'Sector'
        if sector_col in sp500_data.columns:
            print(f"\n--- Available Sectors ---")
            sectors = sp500_data[sector_col].unique()
            for sector in sorted(sectors):
                count = len(sp500_data[sp500_data[sector_col] == sector])
                print(f"{sector}: {count} companies")
        
        # Save to CSV
        save_to_csv(sp500_data)
        
        # Show ticker symbols
        ticker_col = 'Symbol' if 'Symbol' in sp500_data.columns else 'Ticker'
        if ticker_col in sp500_data.columns:
            print(f"\n--- All Ticker Symbols ---")
            tickers = sp500_data[ticker_col].tolist()
            print(f"Total tickers: {len(tickers)}")
            print("First 20 tickers:", ', '.join(tickers[:20]))
    else:
        print("\n❌ All methods failed. Please check your internet connection.")
        print("\nAlternative: Install yfinance library:")
        print("pip install yfinance")
        print("\nThen use: import yfinance as yf; sp500 = yf.Ticker('^GSPC')")