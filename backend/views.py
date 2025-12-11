import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings


from rest_framework.decorators import api_view
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

# ---------- CONFIG & CONSTANTS ----------
# Define these globally or inside the view if you want them dynamic
WINDOW = 50  # Lookback window (days) for time-series slices
BATCH_SIZE = 16  # Mini-batch size for training
EPOCHS = 100  # Max training epochs before early stopping
LR = 1e-3  # Learning rate for Adam optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Prefer GPU if available
SEED = 42  # Random seed for reproducibility
TRAIN_TEST_SPLIT = 0.3  # Test split fraction for holdout
TOP_K_DEFAULT = 10  # Default number of top allocations to return
MAX_ASSETS = 40  # Limit assets to control memory/compute

# Default tuning config (can be overridden via request or Django settings)
TUNING_DEFAULTS = {
    'sector_cap': 0.6,  # Max total weight per sector in final mix
    'momentum_weight': 0.40,  # Blend factor: 1M vs 1W momentum (return tilt)
    'target_vol_annual': 0.12,  # Annualized volatility target for bisection blend
    'recent_days': 300,  # History window for recent metrics (daily/monthly)
    # Risk/Return loss weights (defaults used when payload omits)
    'alpha_return': 2.0,  # Weight for mean portfolio return in training loss
    'beta_risk': 0.5,  # Weight for portfolio risk (std) in training loss
}

# Ensure reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- ML CLASSES (Model, Loss, Dataset) ----------

class AllocNetSmall(nn.Module):
    def __init__(self, num_assets, window, features=3, conv_ch=8, hidden=64):
        super().__init__()
        self.conv = nn.Conv2d(features, conv_ch, kernel_size=(3,1)) 
        conv_out_h = window - 2
        flat_size = conv_ch * conv_out_h * num_assets
        self.shrink = nn.Conv2d(conv_ch, conv_ch, kernel_size=(1,1))
        self.fc1 = nn.Linear(flat_size, hidden)
        self.fc2 = nn.Linear(hidden, num_assets)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        z = torch.relu(self.conv(x))
        z = torch.relu(self.shrink(z))
        z = z.view(z.size(0), -1)
        z = self.dropout(torch.relu(self.fc1(z)))
        w_raw = self.fc2(z)
        w = torch.softmax(w_raw, dim=1)
        return w

class TensorDatasetOnCPU(data.Dataset):
    def __init__(self, X_np, Y_np):
        self.X = X_np.astype(np.float32)
        self.Y = Y_np.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LossFunctions:
    @staticmethod
    def explicit_log_return(weights, price_change_vector, prev_weights, commission=0.0025):
        if price_change_vector.min() < 0.5:
            y_factors = price_change_vector + 1.0
        else:
            y_factors = price_change_vector

        step_factor = torch.sum(weights * y_factors, dim=1)
        turnover = torch.sum(torch.abs(weights - prev_weights), dim=1)
        transaction_cost = commission * turnover
        net_factor = step_factor * (1.0 - transaction_cost)
        net_factor_clamped = torch.clamp(net_factor, min=1e-8)
        log_return = torch.log(net_factor_clamped)
        loss = -torch.mean(log_return)

        if torch.isnan(loss):
            return torch.tensor(1e6, device=weights.device, dtype=weights.dtype)
        return loss

# ---------- Enhanced Model & Loss (matching training_div.py) ----------
class AllocNetWithSectors(nn.Module):
    def __init__(self, num_assets, window, num_features, conv_ch=12, hidden=96):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, conv_ch, kernel_size=(3,1), padding=(1,0))
        self.bn1 = nn.BatchNorm2d(conv_ch)
        self.conv2 = nn.Conv2d(conv_ch, conv_ch*2, kernel_size=(3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(conv_ch*2)
        self.pool = nn.AdaptiveAvgPool2d((1, num_assets))
        flat_size = conv_ch * 2 * num_assets
        self.fc1 = nn.Linear(flat_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.fc3 = nn.Linear(hidden//2, num_assets)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)

    def forward(self, x):
        z = torch.relu(self.bn1(self.conv1(x)))
        z = torch.relu(self.bn2(self.conv2(z)))
        z = self.pool(z)
        z = z.view(z.size(0), -1)
        z = self.dropout1(torch.relu(self.fc1(z)))
        z = self.dropout2(torch.relu(self.fc2(z)))
        w_raw = self.fc3(z)
        w = torch.softmax(w_raw, dim=1)
        return w

class PortfolioLoss:
    @staticmethod
    def combined_loss(weights, returns, prev_weights=None, commission=0.0025, alpha_return=2.0, beta_risk=0.5, use_turnover=False, use_diversity=False):
        portfolio_returns = torch.sum(weights * returns, dim=1)
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-6
        sharpe = mean_return / std_return if std_return.item() > 1e-12 else torch.tensor(0.0, device=returns.device)
        turnover = torch.sum(torch.abs(weights - (prev_weights if prev_weights is not None else weights)), dim=1).mean()
        transaction_cost = 0.001 * turnover if use_turnover else 0.0
        concentration = torch.sum(weights ** 2, dim=1).mean()
        concentration_penalty = 0.1 * concentration
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
        diversity_bonus = (0.01 * entropy) if use_diversity else 0.0
        # Explicit risk-return tradeoff: minimize risk (std) and maximize return (mean)
        risk_term = std_return
        return_term = mean_return
        loss_core = beta_risk * risk_term - alpha_return * return_term
        # Preserve prior regularization terms
        # Optional: add turnover cost and diversity bonus
        loss = loss_core + transaction_cost - diversity_bonus
        return loss, sharpe.item(), return_term.item(), risk_term.item()

# ---------- HELPER: Data Processing ----------

def prepare_data(df, requested_tickers, window):
    # Filter for requested tickers
    df = df[df['Ticker'].isin(requested_tickers)]
    
    if df.empty:
        raise ValueError("No data found for requested tickers.")

    # Sort and pivot
    TICKERS = df["Ticker"].unique().tolist()
    TICKERS.sort()
    
    if len(TICKERS) < 2:
        raise ValueError(f"Need at least 2 assets to allocate, found {len(TICKERS)}.")

    def pivot_feature(df, feature):
        p = df.pivot(index="Date", columns="Ticker", values=feature)
        p = p.sort_index()
        p = p[TICKERS]
        return p

    close_mat = pivot_feature(df, "Close").ffill().bfill()
    high_mat = pivot_feature(df, "High").ffill().bfill()
    low_mat = pivot_feature(df, "Low").ffill().bfill()
    
    # Drop NaNs
    mask = close_mat.isnull().any(axis=1)
    close_mat = close_mat[~mask]
    high_mat = high_mat[~mask]
    low_mat = low_mat[~mask]

    if len(close_mat) < window + 1:
        raise ValueError("Not enough historical data for the window size.")

    dates = close_mat.index.to_numpy()
    num_dates = len(dates)
    
    X_list = []
    Y_list = []

    for i in range(window, num_dates - 1):
        sl = slice(i - window, i)
        c = close_mat.iloc[sl].to_numpy()
        h = high_mat.iloc[sl].to_numpy()
        l = low_mat.iloc[sl].to_numpy()

        last_close = c[-1, :]
        last_close[last_close == 0] = 1.0 

        X = np.stack([l/last_close, h/last_close, c/last_close], axis=0)
        
        p_current = close_mat.iloc[i].to_numpy()
        r_next = (p_current / last_close) - 1.0
        
        X_list.append(X)
        Y_list.append(r_next)

    X_array = np.stack(X_list, axis=0)
    Y_array = np.stack(Y_list, axis=0)
    
    # Prepare last window for inference
    c_last = close_mat.iloc[-window:].to_numpy()
    h_last = high_mat.iloc[-window:].to_numpy()
    l_last = low_mat.iloc[-window:].to_numpy()
    final_close = c_last[-1, :]
    
    X_final = np.stack([l_last/final_close, h_last/final_close, c_last/final_close], axis=0)
    
    return X_array, Y_array, X_final, TICKERS

# ---------- Sector helpers (mapper-driven) ----------
def sector_of(mapper, ticker):
    val = mapper.get(ticker)
    if isinstance(val, dict):
        return val.get('sector', 'Other')
    if isinstance(val, str):
        return val
    return 'Other'

def handle_missing_data(close_mat, high_mat, low_mat, tickers):
    missing_counts = close_mat.isnull().sum()
    missing_pct = (missing_counts / len(close_mat)) * 100
    MISSING_THRESHOLD = 10.0
    assets_to_keep = missing_pct[missing_pct <= MISSING_THRESHOLD].index.tolist()
    if len(assets_to_keep) < len(tickers):
        close_mat = close_mat[assets_to_keep]
        high_mat = high_mat[assets_to_keep]
        low_mat = low_mat[assets_to_keep]
        tickers = assets_to_keep
    close_mat = close_mat.interpolate(method='linear', limit=3, limit_direction='both').ffill().bfill()
    high_mat = high_mat.interpolate(method='linear', limit=3, limit_direction='both').ffill().bfill()
    low_mat = low_mat.interpolate(method='linear', limit=3, limit_direction='both').ffill().bfill()
    mask = close_mat.isnull().any(axis=1) | high_mat.isnull().any(axis=1) | low_mat.isnull().any(axis=1)
    if mask.any():
        close_mat = close_mat[~mask]
        high_mat = high_mat[~mask]
        low_mat = low_mat[~mask]
    return close_mat, high_mat, low_mat, tickers

def build_enhanced_dataset_with_sectors(df, tickers, mapper, window, max_assets=MAX_ASSETS):
    # Memory-safe asset selection
    tickers = sorted(set(tickers))
    if len(tickers) > max_assets:
        if 'Volume' in df.columns:
            avg_vol = df[df['Ticker'].isin(tickers)].groupby('Ticker')['Volume'].mean()
            top_tickers = avg_vol.nlargest(max_assets).index.tolist()
            tickers = sorted(top_tickers)
            df = df[df['Ticker'].isin(tickers)]
        else:
            tickers = tickers[:max_assets]
            df = df[df['Ticker'].isin(tickers)]

    def pivot_feature_local(df_local, feature):
        p = df_local.pivot(index='Date', columns='Ticker', values=feature)
        p = p.sort_index()
        p = p[tickers]
        return p

    close_mat = pivot_feature_local(df, 'Close')
    high_mat = pivot_feature_local(df, 'High')
    low_mat = pivot_feature_local(df, 'Low')
    common_dates = close_mat.index.intersection(high_mat.index).intersection(low_mat.index)
    close_mat = close_mat.loc[common_dates]
    high_mat = high_mat.loc[common_dates]
    low_mat = low_mat.loc[common_dates]

    close_mat, high_mat, low_mat, tickers = handle_missing_data(close_mat, high_mat, low_mat, tickers)
    if len(close_mat) < window + 1:
        raise ValueError(f"Not enough data! Need > {window} rows, but got {len(close_mat)}.")

    # Sector encoding
    sectors = list(set(sector_of(mapper, t) for t in tickers))
    sectors.sort()
    sector_to_idx = {s: i for i, s in enumerate(sectors)}
    sector_matrix = np.zeros((len(tickers), len(sectors)))
    for i, t in enumerate(tickers):
        s = sector_of(mapper, t)
        sector_matrix[i, sector_to_idx[s]] = 1.0

    dates = close_mat.index.to_numpy()
    num_dates = len(dates)
    X_list, Y_list = [], []
    for i in range(window, num_dates - 1):
        sl = slice(i - window, i)
        c = close_mat.iloc[sl].to_numpy()
        h = high_mat.iloc[sl].to_numpy()
        l = low_mat.iloc[sl].to_numpy()
        last_close = c[-1, :]
        last_close[last_close == 0] = 1.0
        c_norm = c / last_close
        h_norm = h / last_close
        l_norm = l / last_close
        price_features = np.stack([l_norm, h_norm, c_norm], axis=0)
        sector_features = sector_matrix.T[:, np.newaxis, :]
        sector_features = np.repeat(sector_features, window, axis=1)
        X = np.concatenate([price_features, sector_features], axis=0)
        p_current = close_mat.iloc[i].to_numpy()
        r_next = (p_current / last_close) - 1.0
        X_list.append(X)
        Y_list.append(r_next)
    X_array = np.stack(X_list, axis=0)
    Y_array = np.stack(Y_list, axis=0)

    # Final window
    c_last = close_mat.iloc[-window:].to_numpy()
    h_last = high_mat.iloc[-window:].to_numpy()
    l_last = low_mat.iloc[-window:].to_numpy()
    final_close = c_last[-1, :]
    c_norm_last = c_last / final_close
    h_norm_last = h_last / final_close
    l_norm_last = l_last / final_close
    price_features_last = np.stack([l_norm_last, h_norm_last, c_norm_last], axis=0)
    sector_features_last = sector_matrix.T[:, np.newaxis, :]
    sector_features_last = np.repeat(sector_features_last, window, axis=1)
    X_final = np.concatenate([price_features_last, sector_features_last], axis=0)
    return X_array, Y_array, X_final, tickers, len(sectors)

# ---------- HELPER: Mapper (Ticker → GICS Sector) ----------
def load_or_build_mapper():
    """
    Loads mapper.json from Training/ if present. Otherwise builds it from
    Training/sp500_companies.csv (columns: Symbol, GICS Sector) and saves mapper.json.
    Returns a dict mapping ticker symbol to GICS Sector.
    """
    try:
        base_dir = getattr(settings, 'BASE_DIR', os.getcwd())
        mapper_path = os.path.join(base_dir, 'Training', 'mapper.json')
        if os.path.exists(mapper_path):
            with open(mapper_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Build mapper from S&P 500 companies CSV
        csv_path = os.path.join(base_dir, 'Training', 'sp500_companies.csv')
        if not os.path.exists(csv_path):
            # Fallback to root if Training folder isn't resolved
            csv_path = os.path.join(base_dir, 'sp500_companies.csv')
        if not os.path.exists(csv_path):
            return {}

        df_map = pd.read_csv(csv_path)
        # Expect columns: Symbol, GICS Sector, optionally Security
        cols = {c.lower(): c for c in df_map.columns}
        sym_col = cols.get('symbol') or cols.get('ticker')
        sec_col = cols.get('gics sector') or cols.get('sector') 
        abbr_col = cols.get('security')
        if not sym_col or not sec_col:
            return {}
        mapper = {}
        use_cols = [sym_col, sec_col] + ([abbr_col] if abbr_col else [])
        for _, row in df_map[use_cols].dropna().iterrows():
            sym = str(row[sym_col]).strip()
            sec = str(row[sec_col]).strip()
            abbr = str(row[abbr_col]).strip() if abbr_col else ''
            if sym:
                mapper[sym] = {'sector': sec, 'abbr': abbr}

        # Persist mapper for future reuse
        try:
            os.makedirs(os.path.dirname(mapper_path), exist_ok=True)
            with open(mapper_path, 'w', encoding='utf-8') as f:
                json.dump(mapper, f)
        except Exception:
            pass
        return mapper
    except Exception:
        return {}

# ---------- DJANGO VIEWS ----------
@api_view(['GET'])
def test(request):
    return JsonResponse({'status': 'API is working'})

response_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'tickers': openapi.Schema(
            type=openapi.TYPE_ARRAY, 
            items=openapi.Schema(type=openapi.TYPE_STRING),
            description="List of available stock tickers",
            example=["AAPL", "MSFT", "GOOG"]
        ),
    }
)

@swagger_auto_schema(
    method='get',
    operation_description="Returns list of available tickers from the dataset",
    responses={200: response_schema, 500: 'Server Error'} 
)
@api_view(['GET'])
def list_tickers(request):
    csv_path = "Training\\diversified_market_data.csv"
    if not os.path.exists(csv_path):
         csv_path = "stocks_market_data.csv" # fallback
    
    if not os.path.exists(csv_path):
        return JsonResponse({'error': 'Data file not found on server'}, status=500)

    df = pd.read_csv(csv_path)
    tickers = df['Ticker'].unique().tolist()
    tickers.sort()
    return JsonResponse({'tickers': tickers})


# Define the Input Schema for Swagger
# This tells Swagger: "I expect a JSON object with a list of strings called 'tickers'"
train_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'tickers': openapi.Schema(
            type=openapi.TYPE_ARRAY, 
            items=openapi.Schema(type=openapi.TYPE_STRING),
            description="List of stock tickers to train on",
            example=["AAPL", "MSFT", "GOOG"]
        ),
    },
    required=['tickers']
)

@swagger_auto_schema(
    method='post',
    request_body=train_schema,
    operation_description="Train the allocation model on specific tickers",
    responses={200: 'Model training started successfully', 400: 'Bad Request'}
)
@api_view(['POST'])
def train_model(request):
    """
    POST Request Body: {"tickers": ["AAPL", "MSFT", "GOOG"]}
    """
    # Note: With @api_view, we use request.data instead of json.loads(request.body)
    try:
        # 1. Parse Input
        target_tickers = request.data.get('tickers', [])
        
        if not target_tickers:
            return Response({'error': 'No tickers provided'}, status=400)

        # 2. Load CSV 
        # (Make sure this path is correct relative to your manage.py)
        csv_path = "diversified_market_data.csv"
        if not os.path.exists(csv_path):
             csv_path = "stocks_market_data.csv"
        
        if not os.path.exists(csv_path):
            return Response({'error': 'Data file not found on server'}, status=500)

        df = pd.read_csv(csv_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # 3. Process Data
        # (Assuming prepare_data, TICKERS, and WINDOW are defined as in previous step)
        try:
            X_array, Y_array, X_final, TICKERS = prepare_data(df, target_tickers, WINDOW=50)
        except ValueError as e:
            return Response({'error': str(e)}, status=400)

        # ... [Keep Steps 4-7 (Model Training Logic) exactly the same] ...
        # For brevity, I am skipping the training loop code here, 
        # but you should keep the logic we wrote in the previous step.

        # 8. Format Output
        results = []
        # (Assuming future_weights was calculated in the training block)
        # Mocking result for syntax correctness if you paste this directly:
        # results = [{"ticker": "AAPL", "allocation": 50.0}, {"ticker": "MSFT", "allocation": 50.0}]

        return Response({
            'status': 'success',
            'tickers_processed': len(TICKERS),
            'portfolio': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)


# ---------- NEW: Train By Sectors Endpoint ----------

train_sectors_schema = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    properties={
        'sectors': openapi.Schema(
            type=openapi.TYPE_ARRAY,
            items=openapi.Schema(type=openapi.TYPE_STRING),
            description='Sector names to include (e.g., Technology, Finance)'
        ),
        'top_k': openapi.Schema(type=openapi.TYPE_INTEGER, description='Number of top allocations to return', default=10),
        'window': openapi.Schema(type=openapi.TYPE_INTEGER, description='Lookback window', default=WINDOW),
        'epochs': openapi.Schema(type=openapi.TYPE_INTEGER, description='Training epochs (small)', default=EPOCHS),
    },
    required=['sectors']
)

@swagger_auto_schema(
    method='post',
    request_body=train_sectors_schema,
    operation_description='Train the allocation model filtered by sectors and return top allocations',
    responses={200: 'Success', 400: 'Bad Request'}
)
@api_view(['POST'])
def train_by_sectors(request):
    try:
        comm_assets_used = []
        # New payload support: {"US Stock": [tickers], "Commodities": [commodity types]}
        us_list = request.data.get('US Stock')
        com_list = request.data.get('Commodities')
        sectors = request.data.get('sectors')

        # Allow requesting all allocations via top_k='all' or <=0
        top_k_param = request.data.get('top_k', TOP_K_DEFAULT)
        try:
            if isinstance(top_k_param, str) and top_k_param.strip().lower() == 'all':
                top_k = -1
            else:
                top_k = int(top_k_param)
        except Exception:
            top_k = TOP_K_DEFAULT
        window = int(request.data.get('window', WINDOW))
        epochs = int(request.data.get('epochs', EPOCHS))
        # New tuning knobs (optional)
        # Load defaults from settings if provided, else use module defaults
        cfg = getattr(settings, 'PORTFOLIO_TUNING_DEFAULTS', TUNING_DEFAULTS)
        sector_cap_req = request.data.get('sector_cap', cfg.get('sector_cap', TUNING_DEFAULTS['sector_cap']))
        momentum_weight_req = request.data.get('momentum_weight', cfg.get('momentum_weight', TUNING_DEFAULTS['momentum_weight']))
        target_vol_annual_req = request.data.get('target_vol_annual', cfg.get('target_vol_annual', TUNING_DEFAULTS['target_vol_annual']))
        recent_days_req = request.data.get('recent_days', cfg.get('recent_days', TUNING_DEFAULTS['recent_days']))
        try:
            sector_cap = float(sector_cap_req)
        except Exception:
            sector_cap = 0.30
        try:
            momentum_weight = float(momentum_weight_req)
            momentum_weight = max(0.0, min(1.0, momentum_weight))
        except Exception:
            momentum_weight = 0.50
        try:
            target_vol_annual = float(target_vol_annual_req) if target_vol_annual_req is not None else None
            if target_vol_annual is not None:
                target_vol_annual = max(0.02, min(0.40, target_vol_annual))
        except Exception:
            target_vol_annual = None
        try:
            recent_days = int(recent_days_req)
            recent_days = max(120, min(1000, recent_days))
        except Exception:
            recent_days = 250
        # New: explicit risk/return tradeoff weights
        alpha_return_req = request.data.get('alpha_return', cfg.get('alpha_return', TUNING_DEFAULTS['alpha_return']))
        beta_risk_req = request.data.get('beta_risk', cfg.get('beta_risk', TUNING_DEFAULTS['beta_risk']))
        try:
            alpha_return = float(alpha_return_req)
        except Exception:
            alpha_return = TUNING_DEFAULTS['alpha_return']
        try:
            beta_risk = float(beta_risk_req)
        except Exception:
            beta_risk = TUNING_DEFAULTS['beta_risk']
        # New flags: enable/disable regularization terms and vol targeting
        def _as_bool(v, default=False):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ('true','1','yes','y','on'): return True
                if s in ('false','0','no','n','off'): return False
            return default
        use_turnover_flag = _as_bool(request.data.get('use_turnover'), False)
        use_diversity_flag = _as_bool(request.data.get('use_diversity'), False)
        disable_vol_targeting = _as_bool(request.data.get('disable_vol_targeting'), False)

        base_dir = getattr(settings, 'BASE_DIR', os.getcwd())
        stock_csv = os.path.join(base_dir, 'Training', 'diversified_market_data.csv')
        comm_csv = os.path.join(base_dir, 'Training', 'Commodities', 'commodities.csv')

        if us_list is not None or com_list is not None:
            # Combined mode: merge stocks and commodities per lists (explicit inclusion only)
            include_stocks = (isinstance(us_list, list) and len(us_list) > 0) or (isinstance(sectors, list) and len(sectors) > 0)
            include_comms = isinstance(com_list, list) and len(com_list) > 0

            # Load stocks if included
            if include_stocks:
                if not os.path.exists(stock_csv):
                    return Response({'error': 'US Stock data file not found on server'}, status=500)
                df_stock = pd.read_csv(stock_csv)
                if 'Date' in df_stock.columns:
                    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
            else:
                df_stock = pd.DataFrame(columns=['Date','Ticker','Open','High','Low','Close','Volume'])

            # Load commodities only if explicitly provided (non-empty list)
            if include_comms:
                if not os.path.exists(comm_csv):
                    return Response({'error': 'Commodities data file not found on server'}, status=500)
                df_comm = pd.read_csv(comm_csv)
                if 'Date' in df_comm.columns:
                    df_comm['Date'] = pd.to_datetime(df_comm['Date'])
                # Normalize commodities: use 'Commodity Type' as Ticker
                if 'Commodity Type' not in df_comm.columns:
                    return Response({'error': 'Commodities CSV missing "Commodity Type" column'}, status=500)
                df_comm = df_comm.rename(columns={'Commodity Type': 'Ticker'})
            else:
                df_comm = pd.DataFrame(columns=['Date','Ticker','Open','High','Low','Close','Volume'])

            # Ensure expected columns exist (Open/High/Low/Close/Volume)
            expected_cols = {'Date','Ticker','Open','High','Low','Close','Volume'}
            missing_stock = expected_cols - set(df_stock.columns)
            missing_comm = expected_cols - set(df_comm.columns)
            if include_stocks and missing_stock:
                return Response({'error': f'US Stock CSV missing columns: {sorted(missing_stock)}'}, status=500)
            if include_comms and missing_comm:
                return Response({'error': f'Commodities CSV missing columns: {sorted(missing_comm)}'}, status=500)

            # Apply filters when including stocks
            if include_stocks:
                if sectors:
                    mapper = load_or_build_mapper() or {}
                    norm = lambda s: str(s).strip().lower()
                    wanted = {norm(s) for s in sectors}
                    def _sector_of(t):
                        val = mapper.get(t)
                        sec = val.get('sector') if isinstance(val, dict) else val
                        return sec
                    tickers_by_sector = [t for t in df_stock['Ticker'].unique().tolist() if _sector_of(t) and norm(_sector_of(t)) in wanted]
                    if tickers_by_sector:
                        df_stock = df_stock[df_stock['Ticker'].isin(tickers_by_sector)]
                if isinstance(us_list, list) and len(us_list) > 0:
                    df_stock = df_stock[df_stock['Ticker'].isin(us_list)]

            # Filter commodities strictly to provided list
            if include_comms:
                df_comm = df_comm[df_comm['Ticker'].isin(com_list)]

            # Track commodities used for response context
            comm_assets_used = sorted(set(df_comm['Ticker'].unique().tolist())) if include_comms else []

            # Merge datasets (only non-empty parts will contribute)
            df_all = pd.concat([df_stock, df_comm], ignore_index=True)
            if df_all.empty:
                return Response({'error': 'No data after filtering US Stock and Commodities selections'}, status=400)

            # Build/extend mapper: use existing for stocks, add commodities → sector "Commodities"
            mapper = load_or_build_mapper() or {}
            if include_comms:
                for ct in df_comm['Ticker'].unique().tolist():
                    if ct not in mapper:
                        mapper[ct] = {'sector': 'Commodities', 'abbr': str(ct)[:6]}

            selected_tickers = sorted(set(df_all['Ticker'].unique().tolist()))
            if len(selected_tickers) < 2:
                return Response({'error': f'Need at least 2 assets, found: {selected_tickers}'}, status=400)

            try:
                X_array, Y_array, X_final, TICKERS, num_sectors = build_enhanced_dataset_with_sectors(df_all, selected_tickers, mapper, window, MAX_ASSETS)
                print(f"[train_by_sectors] Prepared enhanced data (combined): X_array={X_array.shape}, Y_array={Y_array.shape}, X_final={X_final.shape}, assets={len(TICKERS)}, sectors={num_sectors}")
                print(f"Dataset created: {X_array.shape} samples")
                total_channels = X_array.shape[1]
                sector_channels = max(0, total_channels - 3)
                print(f"Features: 3 price + {sector_channels} sector = {total_channels} total channels")
            except ValueError as e:
                return Response({'error': str(e)}, status=400)
        else:
            # Backward-compat: sector-based selection using mapper
            if not sectors:
                return Response({'error': 'Provide either sectors or {"US Stock"/"Commodities"} selections'}, status=400)

            if not os.path.exists(stock_csv):
                return Response({'error': 'US Stock data file not found on server'}, status=500)
            df = pd.read_csv(stock_csv)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])

            mapper = load_or_build_mapper()
            print(f"[train_by_sectors] Requested sectors: {sectors}")
            print(f"[train_by_sectors] Mapper entries: {len(mapper)}")
            norm = lambda s: str(s).strip().lower()
            wanted = {norm(s) for s in sectors}
            available_tickers = set(df['Ticker'].unique().tolist())
            selected_tickers = []
            for t in available_tickers:
                val = mapper.get(t)
                sec = val.get('sector') if isinstance(val, dict) else val
                if sec and norm(sec) in wanted:
                    selected_tickers.append(t)
            selected_tickers = sorted(set(selected_tickers))
            if len(selected_tickers) < 2:
                return Response({'error': f'Not enough tickers found for sectors {sectors}. Found: {selected_tickers}'}, status=400)

            try:
                X_array, Y_array, X_final, TICKERS, num_sectors = build_enhanced_dataset_with_sectors(df, selected_tickers, mapper, window, MAX_ASSETS)
                print(f"[train_by_sectors] Prepared enhanced data: X_array={X_array.shape}, Y_array={Y_array.shape}, X_final={X_final.shape}, assets={len(TICKERS)}, sectors={num_sectors}")
                print(f"Dataset created: {X_array.shape} samples")
                total_channels = X_array.shape[1]
                sector_channels = max(0, total_channels - 3)
                print(f"Features: 3 price + {sector_channels} sector = {total_channels} total channels")
            except ValueError as e:
                return Response({'error': str(e)}, status=400)

        # Train/val split
        from sklearn.model_selection import train_test_split as _tts
        X_train, X_test, Y_train, Y_test = _tts(X_array, Y_array, test_size=TRAIN_TEST_SPLIT, shuffle=False)
        val_size = max(1, int(0.15 * len(X_train)))
        X_train_split = X_train[:-val_size]
        Y_train_split = Y_train[:-val_size]
        X_val = X_train[-val_size:]
        Y_val = Y_train[-val_size:]

        # Datasets & loaders
        train_dataset = TensorDatasetOnCPU(X_train_split, Y_train_split)
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

        # Model
        num_assets = len(TICKERS)
        num_features = X_array.shape[1]
        model = AllocNetWithSectors(num_assets=num_assets, window=window, num_features=num_features).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
        params_count = sum(p.numel() for p in model.parameters())
        print(f"Training on device: {DEVICE}")
        print(f"Model parameters: {params_count:,}")
        print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}\n")
        print(f"[train_by_sectors] Device: {DEVICE} | Num assets: {num_assets} | Features: {num_features} | Window: {window} | Epochs: {epochs} | LR: {LR} | Params: {params_count}")

        # Training loop with validation & early stopping
        best_loss = float('inf')
        patience_counter = 0
        EARLY_STOP_PATIENCE = 15
        # Track Sharpe per epoch for both train and validation
        sharpe_history_train = []
        sharpe_history_val = []
        loss_history_train = []
        loss_history_val = []
        for ep in range(epochs):
            model.train()
            epoch_loss = 0.0
            epoch_sharpe = 0.0
            count = 0
            for xb_np, yb_np in train_loader:
                xb = torch.from_numpy(xb_np).float().to(DEVICE) if not isinstance(xb_np, torch.Tensor) else xb_np.float().to(DEVICE)
                yb = torch.from_numpy(yb_np).float().to(DEVICE) if not isinstance(yb_np, torch.Tensor) else yb_np.float().to(DEVICE)
                opt.zero_grad()
                w = model(xb)
                loss, batch_sharpe, batch_return, batch_risk = PortfolioLoss.combined_loss(
                    w, yb, prev_weights=w.detach(), alpha_return=alpha_return, beta_risk=beta_risk,
                    use_turnover=use_turnover_flag, use_diversity=use_diversity_flag
                )
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
                epoch_sharpe += batch_sharpe * xb.size(0)
                count += xb.size(0)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            avg_train_loss = epoch_loss / max(1, count)
            avg_train_sharpe = epoch_sharpe / max(1, count)
            loss_history_train.append(float(avg_train_loss))
            sharpe_history_train.append(float(avg_train_sharpe))

            # Validation
            model.eval()
            with torch.no_grad():
                X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
                Y_val_t = torch.from_numpy(Y_val).float().to(DEVICE)
                w_val = model(X_val_t)
                val_loss, val_sharpe, val_return, val_risk = PortfolioLoss.combined_loss(
                    w_val, Y_val_t, prev_weights=w_val.detach(), alpha_return=alpha_return, beta_risk=beta_risk,
                    use_turnover=use_turnover_flag, use_diversity=use_diversity_flag
                )
                avg_val_loss = val_loss.item()
                # Record validation metrics for this epoch
                loss_history_val.append(float(avg_val_loss))
                sharpe_history_val.append(float(val_sharpe))
            scheduler.step()

            improved = "✓" if avg_val_loss < best_loss else ""
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"Epoch {ep+1:2d}/{epochs} | Loss: {avg_train_loss:8.4f} | Val: {avg_val_loss:8.4f} | Sharpe: {avg_train_sharpe:6.3f} | Val Sharpe: {val_sharpe:6.3f} {improved}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n⚠️  Early stopping at epoch {ep+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

        # Save Epochs vs Sharpe/Loss plots to outputs folder and CSV
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend for server
            import matplotlib.pyplot as plt
            base_dir = getattr(settings, 'BASE_DIR', os.getcwd())
            outputs_dir = os.path.join(base_dir, 'outputs')
            os.makedirs(outputs_dir, exist_ok=True)
            # Combined Sharpe plot (train vs val)
            fig_path = os.path.join(outputs_dir, 'epochs_vs_sharpe.png')
            plt.figure(figsize=(9, 5))
            epochs_axis = range(1, len(sharpe_history_val) + 1)
            plt.plot(epochs_axis, sharpe_history_train[:len(epochs_axis)], label='Train Sharpe', alpha=0.6)
            plt.plot(epochs_axis, sharpe_history_val, label='Val Sharpe', marker='o', linewidth=2)
            plt.title('Training Epochs vs Sharpe (Train vs Validation)')
            plt.xlabel('Epoch')
            plt.ylabel('Sharpe Ratio')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()

            # Smoothed Sharpe (EMA on validation)
            try:
                import numpy as _np
                ema_alpha = 0.2
                val_arr = _np.array(sharpe_history_val, dtype=float)
                ema = _np.zeros_like(val_arr)
                if len(val_arr) > 0:
                    ema[0] = val_arr[0]
                    for i in range(1, len(val_arr)):
                        ema[i] = ema_alpha * val_arr[i] + (1 - ema_alpha) * ema[i-1]
                fig_path_smoothed = os.path.join(outputs_dir, 'epochs_vs_sharpe_smoothed.png')
                plt.figure(figsize=(9, 5))
                plt.plot(epochs_axis, val_arr, label='Val Sharpe', color='#FF7F0E', alpha=0.35)
                plt.plot(epochs_axis, ema, label='Val Sharpe (EMA 0.2)', color='#FF7F0E', linewidth=2)
                plt.title('Validation Sharpe (EMA Smoothed)')
                plt.xlabel('Epoch')
                plt.ylabel('Sharpe Ratio')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(fig_path_smoothed, dpi=150)
                plt.close()
            except Exception:
                fig_path_smoothed = None

            # Loss plot (train vs val)
            loss_fig_path = os.path.join(outputs_dir, 'epochs_vs_loss.png')
            plt.figure(figsize=(9, 5))
            plt.plot(epochs_axis, loss_history_train[:len(epochs_axis)], label='Train Loss', alpha=0.6)
            plt.plot(epochs_axis, loss_history_val, label='Val Loss', marker='o', linewidth=2)
            plt.title('Training Epochs vs Loss (Train vs Validation)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(loss_fig_path, dpi=150)
            plt.close()

            # Save raw history to CSV for deeper analysis
            hist_csv_path = os.path.join(outputs_dir, 'training_history.csv')
            hist_df = pd.DataFrame({
                'epoch': list(epochs_axis),
                'train_sharpe': sharpe_history_train[:len(epochs_axis)],
                'val_sharpe': sharpe_history_val,
                'train_loss': loss_history_train[:len(epochs_axis)],
                'val_loss': loss_history_val,
            })
            hist_df.to_csv(hist_csv_path, index=False)
        except Exception as e:
            fig_path = None
            loss_fig_path = None
            hist_csv_path = None
            print(f"[train_by_sectors] Failed to save Sharpe plot: {e}")

        # Inference for next-day allocation
        model.eval()
        with torch.no_grad():
            Xf = torch.from_numpy(X_final).float().unsqueeze(0).to(DEVICE)
            future_weights = model(Xf).cpu().numpy().flatten()
        print(f"[train_by_sectors] Inference weights: min={future_weights.min():.6f} max={future_weights.max():.6f} sum={future_weights.sum():.6f}")

        # --- Balanced Post-Processing using evaluation metrics ---
        # Build recent returns matrices to compute momentum, volatility and covariance
        try:
            def _pivot_close(df_local):
                p = df_local.pivot(index='Date', columns='Ticker', values='Close').sort_index()
                return p[TICKERS].ffill().bfill()

            close_mat_all = _pivot_close(df_all if 'df_all' in locals() else df)
            # Limit to a reasonable recent window for metrics
            close_recent = close_mat_all.tail(recent_days)
            daily_ret = close_recent.pct_change().dropna()
            # Monthly returns for covariance-based risk
            monthly_close = close_mat_all.resample('ME').last().dropna()
            monthly_ret = monthly_close.pct_change().dropna()
            cov_m = monthly_ret.cov().values

            # Per-asset daily vol and momentum
            vol_d = daily_ret.std().values + 1e-12
            mom_1m = (1.0 + daily_ret.tail(21)).prod().values - 1.0
            mom_1w = (1.0 + daily_ret.tail(5)).prod().values - 1.0

            # Candidate portfolios
            inv_vol = (1.0 / vol_d)
            inv_vol = np.clip(inv_vol, 0, np.percentile(inv_vol, 95))
            inv_vol = inv_vol / inv_vol.sum()

            mom_blend = momentum_weight * np.maximum(mom_1m, 0) + (1.0 - momentum_weight) * np.maximum(mom_1w, 0)
            if mom_blend.sum() <= 1e-12:
                mom_blend = np.ones_like(mom_blend)
            mom_blend = mom_blend / mom_blend.sum()

            balanced_raw = 0.5 * inv_vol + 0.5 * mom_blend
            balanced_raw = balanced_raw / balanced_raw.sum()

            # Sector caps for diversification
            mapper_local = load_or_build_mapper() or {}
            sectors_of_assets = []
            for t in TICKERS:
                val = mapper_local.get(t, {})
                sec = val.get('sector') if isinstance(val, dict) else (val or 'Other')
                sectors_of_assets.append(sec)

            def apply_sector_caps(w, cap=0.30):
                w = w.copy()
                # Iteratively project onto sector-cap constraints
                for _ in range(5):
                    sector_sum = {}
                    for i, s in enumerate(sectors_of_assets):
                        sector_sum[s] = sector_sum.get(s, 0.0) + w[i]
                    adjusted = False
                    for s, total in sector_sum.items():
                        if total > cap:
                            # Scale down weights in this sector
                            factor = cap / total
                            for i, s_i in enumerate(sectors_of_assets):
                                if s_i == s:
                                    w[i] *= factor
                            adjusted = True
                    if not adjusted:
                        break
                    # Renormalize after adjustments
                    w_sum = w.sum()
                    if w_sum > 1e-12:
                        w = w / w_sum
                return w

            balanced_capped = apply_sector_caps(balanced_raw, cap=sector_cap)

            # Risk ordering vs. inference: aim for moderate annual volatility
            def annual_vol(w, cov):
                return float(np.sqrt(w @ cov @ w) * np.sqrt(12.0))

            w_model = future_weights / future_weights.sum()
            vol_model = annual_vol(w_model, cov_m)
            vol_bal = annual_vol(balanced_capped, cov_m)

            # Blend to target volatility (optional disable)
            if disable_vol_targeting:
                final_weights = w_model
            else:
                if target_vol_annual is None:
                    target_vol = 0.5 * (vol_model + vol_bal)
                else:
                    target_vol = float(target_vol_annual)
                # Bisection-style blend on alpha in [0,1]
                alpha_low, alpha_high = 0.0, 1.0
                final_weights = None
                for _ in range(20):
                    alpha = 0.5 * (alpha_low + alpha_high)
                    w_mix = alpha * w_model + (1 - alpha) * balanced_capped
                    w_mix = w_mix / max(1e-12, w_mix.sum())
                    v = annual_vol(w_mix, cov_m)
                    if v > target_vol:
                        alpha_high = alpha
                    else:
                        alpha_low = alpha
                    final_weights = w_mix
                if final_weights is None:
                    final_weights = balanced_capped

            # Compute evaluation metrics for the final balanced portfolio
            port_m = (monthly_ret.values @ final_weights)
            mean_m = float(np.mean(port_m))
            vol_m = float(np.std(port_m) + 1e-12)
            vol_annual = float(vol_m * np.sqrt(12.0))
            sharpe = float(mean_m / vol_m) if vol_m > 0 else 0.0
            # Sortino (downside std)
            downside = np.std(np.minimum(port_m, 0.0)) + 1e-12
            sortino = float(mean_m / downside) if downside > 0 else 0.0
            # Max Drawdown over monthly equity curve
            eq = np.cumprod(1.0 + port_m)
            peak = np.maximum.accumulate(eq)
            mdd = float(np.max((peak - eq) / peak)) if len(eq) else 0.0
            # CVaR 95%
            q = np.quantile(port_m, 0.05) if len(port_m) else 0.0
            cvar95 = float(np.mean(port_m[port_m <= q])) if len(port_m) else 0.0
            # Short-term daily
            st_1w = float((1.0 + (daily_ret.values @ final_weights)[-5:]).prod() - 1.0) if len(daily_ret) >= 5 else 0.0
            st_1m = float((1.0 + (daily_ret.values @ final_weights)[-21:]).prod() - 1.0) if len(daily_ret) >= 21 else 0.0

            # Rolling 12m backtest metrics (guard against NaNs)
            try:
                # Use last 120 months if available
                roll_window = 12
                port_m_series = pd.Series(port_m)
                # Rolling Sharpe (mean/std in window)
                roll_mean = port_m_series.rolling(roll_window, min_periods=roll_window).mean()
                roll_std = port_m_series.rolling(roll_window, min_periods=roll_window).std() + 1e-12
                rs_val = (roll_mean / roll_std).iloc[-1] if len(port_m_series) >= roll_window else 0.0
                rolling_sharpe_12m = float(rs_val) if np.isfinite(rs_val) else 0.0
                # Rolling max drawdown over last 12m
                window_slice = port_m_series.iloc[-roll_window:] if len(port_m_series) >= roll_window else port_m_series
                eq_w = np.cumprod(1.0 + window_slice.values)
                peak_w = np.maximum.accumulate(eq_w)
                mdd_val = np.max((peak_w - eq_w) / peak_w) if len(eq_w) else 0.0
                rolling_mdd_12m = float(mdd_val) if np.isfinite(mdd_val) else 0.0
            except Exception:
                rolling_sharpe_12m = 0.0
                rolling_mdd_12m = 0.0

            # Turnover proxy vs model weights (one-step difference)
            avg_turnover = float(np.sum(np.abs(final_weights - (future_weights / future_weights.sum()))))
        except Exception as e:
            # Fallback to model weights if metrics fail
            print(f"[train_by_sectors] Balanced post-processing failed: {e}")
            final_weights = future_weights / future_weights.sum()
            mean_m = vol_m = vol_annual = sharpe = sortino = mdd = cvar95 = st_1w = st_1m = 0.0
            rolling_sharpe_12m = rolling_mdd_12m = avg_turnover = 0.0

        # Top-K allocations
        # Determine indices for top allocations
        if isinstance(top_k, int) and top_k > 0:
            idxs = np.argsort(final_weights)[-top_k:][::-1]
        else:
            # 'all' or non-positive -> return all assets sorted
            idxs = np.argsort(final_weights)[::-1]
        portfolio = []
        for i in idxs:
            tkr = TICKERS[i]
            val = mapper.get(tkr, {})
            sector = val.get('sector') if isinstance(val, dict) else (val or 'Other')
            abbr = val.get('abbr') if isinstance(val, dict) else ''
            portfolio.append({
                'ticker': tkr,
                'allocation': float(final_weights[i] * 100.0),
                'sector': sector,
                'abbr': abbr
            })
        print("[train_by_sectors] Top allocations:")
        for r in portfolio:
            print(f"  {r.get('abbr') or r['ticker']:>20s} | {r['sector']:<24s} | {r['allocation']:6.2f}%")

        # Sector distribution
        from collections import defaultdict
        sector_alloc = defaultdict(float)
        for r in portfolio:
            sector_alloc[r['sector']] += r['allocation']
        print("[train_by_sectors] Sector Distribution:")
        for s, a in sorted(sector_alloc.items(), key=lambda x: -x[1]):
            print(f"  {s:24s}: {a:6.2f}%")

            # Ensure sectors field reflects training inputs:
            # - If commodities were used, include 'Commodities' alongside any provided stock sectors
        out_sectors = sectors if sectors else []
        try:
            req_com = request.data.get('Commodities')
            if isinstance(req_com, list) and len(req_com) > 0:
                if 'Commodities' not in out_sectors:
                    out_sectors = list(out_sectors) + ['Commodities']
        except Exception:
            pass

        return Response({
            'status': 'success',
            'sectors': out_sectors,
            'tickers_used': TICKERS,
            'portfolio': portfolio,
            'assets_used': comm_assets_used if 'comm_assets_used' in locals() else [],
            'outputs': {
                'epochs_vs_sharpe_path': ('outputs/epochs_vs_sharpe.png' if 'fig_path' in locals() and fig_path else None),
                'epochs_vs_loss_path': ('outputs/epochs_vs_loss.png' if 'loss_fig_path' in locals() and loss_fig_path else None),
                'training_history_csv': ('outputs/training_history.csv' if 'hist_csv_path' in locals() and hist_csv_path else None),
                'epochs_vs_sharpe_smoothed_path': ('outputs/epochs_vs_sharpe_smoothed.png' if 'fig_path_smoothed' in locals() and fig_path_smoothed else None)
            },
            'params_used': {
                'sector_cap': sector_cap,
                'momentum_weight': momentum_weight,
                'target_vol_annual': target_vol_annual,
                'recent_days': recent_days,
                'alpha_return': alpha_return,
                'beta_risk': beta_risk,
                'window': window,
                'epochs': epochs,
                'top_k': top_k if isinstance(top_k, int) else TOP_K_DEFAULT,
                'use_turnover': use_turnover_flag,
                'use_diversity': use_diversity_flag,
                'disable_vol_targeting': disable_vol_targeting
            },
            'metrics': {
                'mean_monthly': mean_m,
                'vol_monthly': vol_m,
                'vol_annual': vol_annual,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_drawdown': mdd,
                'cvar_95': cvar95,
                'short_term_1w': st_1w,
                'short_term_1m': st_1m,
                'rolling_sharpe_12m': rolling_sharpe_12m,
                'rolling_mdd_12m': rolling_mdd_12m,
                'avg_turnover': avg_turnover
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)