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

def test(request):
    return JsonResponse({'status': 'API is working'})

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

@csrf_exempt
def train_model(request):
    """
    POST Request Body: {"tickers": ["AAPL", "MSFT", "GOOG"]}
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        # 1. Parse Input
        body = json.loads(request.body)
        target_tickers = body.get('tickers', [])
        
        if not target_tickers:
            return JsonResponse({'error': 'No tickers provided'}, status=400)

        # 2. Load CSV (Assuming file is in root or specify path)
        csv_path = "Training\\diversified_market_data.csv"
        if not os.path.exists(csv_path):
             csv_path = "stocks_market_data.csv" # fallback
        
        if not os.path.exists(csv_path):
            return JsonResponse({'error': 'Data file not found on server'}, status=500)

        df = pd.read_csv(csv_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # 3. Process Data
        try:
            X_array, Y_array, X_final, TICKERS = prepare_data(df, target_tickers, WINDOW)
        except ValueError as e:
            return JsonResponse({'error': str(e)}, status=400)

        # 4. Train/Test Split
        X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=0.2, shuffle=False)
        
        # 5. Initialize Model
        num_assets = len(TICKERS)
        model = AllocNetSmall(num_assets=num_assets, window=WINDOW).to(DEVICE)
        opt = optim.Adam(model.parameters(), lr=LR)
        
        train_dataset = TensorDatasetOnCPU(X_train, Y_train)
        train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 6. Training Loop (Condensed)
        print(f"Starting training on {TICKERS} for {EPOCHS} epochs...")
        model.train()
        for epoch in range(EPOCHS):
            for xb_np, yb_np in train_loader:
                xb = xb_np.to(DEVICE)
                yb = yb_np.to(DEVICE)
                
                opt.zero_grad()
                w = model(xb)
                yb_factors = 1.0 + yb
                loss = LossFunctions.explicit_log_return(w, yb_factors, w.detach())
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
        
        # 7. Inference (Future Prediction)
        model.eval()
        X_final_t = torch.tensor(X_final, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            future_weights = model(X_final_t).cpu().numpy().flatten()

        # 8. Format Output
        results = []
        for i, ticker in enumerate(TICKERS):
            pct = float(future_weights[i] * 100)
            if pct > 0.01: # Filter tiny values
                results.append({
                    "ticker": ticker,
                    "allocation": round(pct, 2)
                })
        
        # Sort by allocation
        results.sort(key=lambda x: x['allocation'], reverse=True)

        return JsonResponse({
            'status': 'success',
            'tickers_processed': len(TICKERS),
            'epochs_trained': EPOCHS,
            'portfolio': results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)