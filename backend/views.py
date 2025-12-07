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
WINDOW = 50
BATCH_SIZE = 16  # Small batch for API
EPOCHS = 5       # REDUCED for API speed (increase if using Async tasks)
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

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

# Minimal sector map to translate sectors → tickers (aligns with Training/training_div.py)
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'NVDA': 'Technology', 'META': 'Technology', 'TSLA': 'Technology',
    'AMD': 'Technology', 'INTC': 'Technology', 'ORCL': 'Technology',
    'AVGO': 'Technology', 'CRM': 'Technology', 'ADBE': 'Technology',
    # Finance
    'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance',
    'GS': 'Finance', 'MS': 'Finance', 'C': 'Finance',
    'V': 'Finance', 'MA': 'Finance', 'AXP': 'Finance',
    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare',
    'ABBV': 'Healthcare', 'TMO': 'Healthcare', 'MRK': 'Healthcare',
    'LLY': 'Healthcare', 'ABT': 'Healthcare',
    # Consumer
    'AMZN': 'Consumer', 'WMT': 'Consumer', 'HD': 'Consumer',
    'NKE': 'Consumer', 'MCD': 'Consumer', 'SBUX': 'Consumer',
    'COST': 'Consumer', 'TGT': 'Consumer',
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'SLB': 'Energy', 'EOG': 'Energy',
    # Industrials
    'BA': 'Industrials', 'CAT': 'Industrials', 'GE': 'Industrials',
    'UPS': 'Industrials', 'HON': 'Industrials',
    # Index/ETFs
    'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', 'DIA': 'Index',
}

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
        sectors = request.data.get('sectors', [])
        if not sectors:
            return Response({'error': 'No sectors provided'}, status=400)

        top_k = int(request.data.get('top_k', 10))
        window = int(request.data.get('window', WINDOW))
        epochs = int(request.data.get('epochs', EPOCHS))

        # Load CSV
        csv_path = "Training/diversified_market_data.csv"
        if not os.path.exists(csv_path):
            csv_path = "Training/stocks_market_data.csv"
        if not os.path.exists(csv_path):
            return Response({'error': 'Data file not found on server'}, status=500)

        df = pd.read_csv(csv_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # Map sectors to tickers and intersect with available tickers in CSV
        available_tickers = set(df['Ticker'].unique().tolist())
        selected_tickers = [t for t, s in SECTOR_MAP.items() if s in sectors and t in available_tickers]
        selected_tickers = sorted(set(selected_tickers))

        if len(selected_tickers) < 2:
            return Response({'error': f'Not enough tickers found for sectors {sectors}. Found: {selected_tickers}'}, status=400)

        # Prepare data
        try:
            X_array, Y_array, X_final, TICKERS = prepare_data(df, selected_tickers, window)
        except ValueError as e:
            return Response({'error': str(e)}, status=400)

        # Dataset & loader
        ds = TensorDatasetOnCPU(X_array, Y_array)
        loader = data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Model
        num_assets = len(TICKERS)
        model = AllocNetSmall(num_assets=num_assets, window=window).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR)

        # Quick training loop for API
        model.train()
        for ep in range(max(1, epochs)):
            epoch_loss = 0.0
            for xb_np, yb_np in loader:
                # Support both numpy arrays and already-created torch tensors
                if isinstance(xb_np, torch.Tensor):
                    xb = xb_np.float().to(DEVICE)
                    yb = yb_np.float().to(DEVICE)
                else:
                    xb = torch.from_numpy(xb_np).float().to(DEVICE)
                    yb = torch.from_numpy(yb_np).float().to(DEVICE)
                opt.zero_grad()
                w = model(xb)
                loss = LossFunctions.explicit_log_return(w, yb, prev_weights=w.detach())
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                epoch_loss += loss.item()

        # Inference for next-day allocation
        model.eval()
        with torch.no_grad():
            Xf = torch.from_numpy(X_final).float().unsqueeze(0).to(DEVICE)
            future_weights = model(Xf).cpu().numpy().flatten()

        # Top-K allocations
        idxs = np.argsort(future_weights)[-top_k:][::-1]
        portfolio = [
            {
                'ticker': TICKERS[i],
                'allocation': float(future_weights[i] * 100.0),
                'sector': SECTOR_MAP.get(TICKERS[i], 'Other')
            }
            for i in idxs
        ]

        return Response({
            'status': 'success',
            'sectors': sectors,
            'tickers_used': TICKERS,
            'portfolio': portfolio
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response({'error': str(e)}, status=500)