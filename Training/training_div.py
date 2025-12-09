import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import torch.utils.data as data
import json

# ---------- CONFIG ----------
CSV_PATH = "diversified_market_data.csv" 
if not os.path.exists(CSV_PATH):
    CSV_PATH = "stocks_market_data.csv"

WINDOW = 50
TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 32  
EPOCHS = 40
LR = 1e-3  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TOP_K = 10
MAX_ASSETS = 40

np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- MAPPER (Ticker → GICS Sector, Abbreviation) ----------
def load_or_build_mapper():
    """
    Load Training/mapper.json if present; otherwise build it from
    Training/sp500_companies.csv (Symbol, GICS Sector, optionally Security).
    Returns dict: { ticker: { 'sector': str, 'abbr': str } } or {}.
    """
    # Candidate paths
    mapper_paths = [os.path.join('Training', 'mapper.json'), 'mapper.json']
    csv_paths = [os.path.join('Training', 'sp500_companies.csv'), 'sp500_companies.csv']

    # Try loading existing mapper
    for mp in mapper_paths:
        if os.path.exists(mp):
            try:
                with open(mp, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                pass

    # Build mapper from CSV
    csv_path = None
    for cp in csv_paths:
        if os.path.exists(cp):
            csv_path = cp
            break
    if not csv_path:
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    # Detect columns
    cols = {c.lower(): c for c in df.columns}
    sym_col = cols.get('symbol') or cols.get('ticker')
    sec_col = cols.get('gics sector') or cols.get('sector')
    abbr_col = cols.get('security')
    if not sym_col or not sec_col:
        return {}

    mapper = {}
    use_cols = [sym_col, sec_col] + ([abbr_col] if abbr_col else [])
    for _, row in df[use_cols].dropna().iterrows():
        sym = str(row[sym_col]).strip()
        sec = str(row[sec_col]).strip()
        abbr = str(row[abbr_col]).strip() if abbr_col else ''
        if sym:
            mapper[sym] = {'sector': sec, 'abbr': abbr}

    # Persist mapper
    mp = mapper_paths[0]
    try:
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        with open(mp, 'w', encoding='utf-8') as f:
            json.dump(mapper, f, indent=2)
    except Exception:
        pass
    return mapper

MAPPER = load_or_build_mapper()

def sector_of(ticker):
    val = MAPPER.get(ticker)
    if isinstance(val, dict):
        return val.get('sector', 'Other')
    if isinstance(val, str):
        return val
    return 'Other'

# ---------- UTIL: metrics ----------
def portfolio_metrics(portfolio_values, periods_per_year=252):
    rets = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(rets) == 0:
        return {}
    cumulative_return = portfolio_values[-1] - 1.0
    ann_return = (portfolio_values[-1]) ** (periods_per_year / len(rets)) - 1
    ann_vol = np.std(rets) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol != 0 else 0.0
    
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / running_max
    max_dd = drawdowns.max()
    
    return {
        "cum_ret": cumulative_return,
        "ann_ret": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd
    }

# ---------- IMPROVED MISSING DATA HANDLING ----------
def handle_missing_data(close_mat, high_mat, low_mat, tickers):
    """Multi-stage approach to handle missing data robustly"""
    print("\n--- Missing Data Analysis ---")
    
    missing_counts = close_mat.isnull().sum()
    missing_pct = (missing_counts / len(close_mat)) * 100
    
    print(f"Assets with missing data: {(missing_counts > 0).sum()}/{len(tickers)}")
    if (missing_counts > 0).any():
        worst_assets = missing_pct.nlargest(5)
        print("Assets with most missing data:")
        for ticker, pct in worst_assets.items():
            print(f"  {ticker}: {pct:.2f}% missing")
    
    MISSING_THRESHOLD = 10.0
    assets_to_keep = missing_pct[missing_pct <= MISSING_THRESHOLD].index.tolist()
    
    if len(assets_to_keep) < len(tickers):
        dropped = set(tickers) - set(assets_to_keep)
        print(f"\nDropping {len(dropped)} assets with >{MISSING_THRESHOLD}% missing data: {sorted(dropped)}")
        
        close_mat = close_mat[assets_to_keep]
        high_mat = high_mat[assets_to_keep]
        low_mat = low_mat[assets_to_keep]
        tickers = assets_to_keep
    
    close_mat = close_mat.interpolate(method='linear', limit=3, limit_direction='both')
    high_mat = high_mat.interpolate(method='linear', limit=3, limit_direction='both')
    low_mat = low_mat.interpolate(method='linear', limit=3, limit_direction='both')
    
    close_mat = close_mat.ffill().bfill()
    high_mat = high_mat.ffill().bfill()
    low_mat = low_mat.ffill().bfill()
    
    mask = close_mat.isnull().any(axis=1) | high_mat.isnull().any(axis=1) | low_mat.isnull().any(axis=1)
    if mask.any():
        print(f"Dropping {mask.sum()} dates with remaining NaNs")
        close_mat = close_mat[~mask]
        high_mat = high_mat[~mask]
        low_mat = low_mat[~mask]
    
    print(f"Final dataset: {len(close_mat)} dates × {len(tickers)} assets\n")
    
    return close_mat, high_mat, low_mat, tickers

# ---------- STEP 1: Load and Pivot CSV ----------
print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
else:
    raise RuntimeError("CSV missing 'Date' column.")

TICKERS = df["Ticker"].unique().tolist()
TICKERS.sort()
print(f"Detected {len(TICKERS)} tickers initially")

# ---------- MEMORY-SAFE ASSET SELECTION ----------
if len(TICKERS) > MAX_ASSETS:
    print(f"\n{'='*60}")
    print(f"⚠️  MEMORY OPTIMIZATION: Reducing {len(TICKERS)} → {MAX_ASSETS} assets")
    print(f"{'='*60}")
    
    if 'Volume' in df.columns:
        avg_vol = df.groupby('Ticker')['Volume'].mean()
        top_tickers = avg_vol.nlargest(MAX_ASSETS).index.tolist()
        TICKERS = sorted(top_tickers)
        df = df[df['Ticker'].isin(TICKERS)]
        print(f"✓ Selected top {MAX_ASSETS} by trading volume")
    else:
        TICKERS = TICKERS[:MAX_ASSETS]
        df = df[df['Ticker'].isin(TICKERS)]
        print(f"✓ Selected first {MAX_ASSETS} alphabetically")
    
    print(f"Final tickers: {TICKERS}")
    print(f"{'='*60}\n")
else:
    print(f"Using all {len(TICKERS)} tickers\n")

def pivot_feature(df, feature):
    p = df.pivot(index="Date", columns="Ticker", values=feature)
    p = p.sort_index()
    p = p[TICKERS] 
    return p

try:
    close_mat = pivot_feature(df, "Close")
    high_mat  = pivot_feature(df, "High")
    low_mat   = pivot_feature(df, "Low")
except KeyError as e:
    print(f"Error: Column {e} not found in CSV. Check your data headers.")
    exit()

common_dates = close_mat.index.intersection(high_mat.index).intersection(low_mat.index)
close_mat = close_mat.loc[common_dates]
high_mat  = high_mat.loc[common_dates]
low_mat   = low_mat.loc[common_dates]

# ---------- STEP 2: Handle Missing Data ----------
close_mat, high_mat, low_mat, TICKERS = handle_missing_data(close_mat, high_mat, low_mat, TICKERS)

if len(close_mat) < WINDOW + 1:
    raise RuntimeError(f"Not enough data! Need > {WINDOW} rows, but got {len(close_mat)}.")

# ---------- CREATE SECTOR FEATURES ----------
def create_sector_encoding(tickers):
    """Create one-hot encoding for sectors"""
    sectors = list(set(sector_of(t) for t in tickers))
    sectors.sort()
    sector_to_idx = {s: i for i, s in enumerate(sectors)}
    
    print(f"Sectors found: {sectors}")
    
    sector_matrix = np.zeros((len(tickers), len(sectors)))
    for i, ticker in enumerate(tickers):
        sector = sector_of(ticker)
        sector_idx = sector_to_idx[sector]
        sector_matrix[i, sector_idx] = 1.0
    
    return sector_matrix, sectors

sector_encoding, sector_names = create_sector_encoding(TICKERS)
print(f"Sector encoding shape: {sector_encoding.shape}\n")

# ---------- STEP 3: Build Enhanced Tensors with Sector Features ----------
dates = close_mat.index.to_numpy()
num_dates = len(dates)
num_assets = len(TICKERS)

print(f"Building dataset with {num_dates} dates and {num_assets} assets...")

X_list = []
Y_list = []       
y_dates = []

for i in range(WINDOW, num_dates - 1):
    sl = slice(i - WINDOW, i)
    
    c = close_mat.iloc[sl].to_numpy()
    h = high_mat.iloc[sl].to_numpy()
    l = low_mat.iloc[sl].to_numpy()

    last_close = c[-1, :]
    last_close[last_close == 0] = 1.0

    c_norm = c / last_close
    h_norm = h / last_close
    l_norm = l / last_close

    price_features = np.stack([l_norm, h_norm, c_norm], axis=0)
    
    sector_features = sector_encoding.T[:, np.newaxis, :]
    sector_features = np.repeat(sector_features, WINDOW, axis=1)
    
    X = np.concatenate([price_features, sector_features], axis=0)

    p_current = close_mat.iloc[i].to_numpy()
    r_next = (p_current / last_close) - 1.0
    
    X_list.append(X)
    Y_list.append(r_next)
    y_dates.append(dates[i])

X_array = np.stack(X_list, axis=0)
Y_array = np.stack(Y_list, axis=0)

print(f"Dataset created: {X_array.shape} samples")
print(f"Features: 3 price + {len(sector_names)} sector = {X_array.shape[1]} total channels")

# ---------- DATA DIAGNOSTICS ----------
print("\n--- Return Statistics (Training Data Preview) ---")
print(f"Mean return per asset: {Y_array.mean(axis=0).mean()*100:.4f}%")
print(f"Std return per asset: {Y_array.std(axis=0).mean()*100:.4f}%")
print(f"Max single return: {Y_array.max()*100:.2f}%")
print(f"Min single return: {Y_array.min()*100:.2f}%")
print(f"Return variance across time: {Y_array.var(axis=1).mean():.6f}\n")

# ---------- STEP 4: Train/Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=TRAIN_TEST_SPLIT, shuffle=False)

# ---------- STEP 5: Enhanced Neural Net ----------
CONV_CHANNELS = 12
HIDDEN = 96

class AllocNetWithSectors(nn.Module):
    def __init__(self, num_assets, window, num_features, conv_ch=CONV_CHANNELS, hidden=HIDDEN):
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

# ---------- IMPROVED LOSS FUNCTION ----------
class PortfolioLoss:
    @staticmethod
    def combined_loss(weights, returns, prev_weights=None, commission=0.0025):
        """
        Improved loss with better weighting for Sharpe signal
        """
        portfolio_returns = torch.sum(weights * returns, dim=1)
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-6
        
        sharpe = mean_return / std_return
        
        # Reduced penalty weights to let Sharpe dominate
        if prev_weights is not None:
            turnover = torch.sum(torch.abs(weights - prev_weights), dim=1).mean()
            transaction_cost = 0.001 * turnover  # Reduced from 0.0025
        else:
            transaction_cost = 0.0
        
        concentration = torch.sum(weights ** 2, dim=1).mean()
        concentration_penalty = 0.1 * concentration  # Reduced from 0.5
        
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
        diversity_bonus = 0.01 * entropy  # Increased from 0.001
        
        # Give more weight to Sharpe (10x multiplier)
        loss = -10.0 * sharpe + transaction_cost + concentration_penalty - diversity_bonus
        
        return loss, sharpe.item()  # Return actual Sharpe for monitoring

# ---------- Dataset and Training ----------
class TensorDatasetOnCPU(data.Dataset):
    def __init__(self, X_np, Y_np):
        self.X = X_np.astype(np.float32)
        self.Y = Y_np.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Validation split
val_size = int(0.15 * len(X_train))
X_train_split = X_train[:-val_size]
Y_train_split = Y_train[:-val_size]
X_val = X_train[-val_size:]
Y_val = Y_train[-val_size:]

train_dataset = TensorDatasetOnCPU(X_train_split, Y_train_split)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)

# Initialize model
num_features = X_train.shape[1]
model = AllocNetWithSectors(num_assets=num_assets, window=WINDOW, num_features=num_features).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Cosine annealing with warm restarts for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)

def batch_to_device(batch):
    xb, yb = batch
    if isinstance(xb, torch.Tensor):
        xb = xb.float().to(DEVICE)
        yb = yb.float().to(DEVICE)
    else:
        xb = torch.from_numpy(xb).float().to(DEVICE)
        yb = torch.from_numpy(yb).float().to(DEVICE)
    return xb, yb

print(f"Training on device: {DEVICE}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}\n")

best_loss = float('inf')
patience_counter = 0
EARLY_STOP_PATIENCE = 15  # Increased patience

train_sharpes = []
val_sharpes = []

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    epoch_loss = 0.0
    epoch_sharpe = 0.0
    count = 0
    
    for batch in train_loader:
        xb, yb = batch_to_device(batch)
        opt.zero_grad()
        w = model(xb)
        
        loss, batch_sharpe = PortfolioLoss.combined_loss(w, yb, prev_weights=w.detach())
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf loss encountered; skipping batch")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        epoch_loss += loss.item() * xb.size(0)
        epoch_sharpe += batch_sharpe * xb.size(0)
        count += xb.size(0)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    avg_train_loss = epoch_loss / count
    avg_train_sharpe = epoch_sharpe / count
    train_sharpes.append(avg_train_sharpe)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
        Y_val_t = torch.from_numpy(Y_val).float().to(DEVICE)
        w_val = model(X_val_t)
        val_loss, val_sharpe = PortfolioLoss.combined_loss(w_val, Y_val_t, prev_weights=w_val.detach())
        avg_val_loss = val_loss.item()
        val_sharpes.append(val_sharpe)
    
    scheduler.step()
    
    # Early stopping based on validation loss
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        improvement = "✓"
    else:
        patience_counter += 1
        improvement = ""
    
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_train_loss:8.4f} | Val: {avg_val_loss:8.4f} | "
              f"Sharpe: {avg_train_sharpe:6.3f} | Val Sharpe: {val_sharpe:6.3f} {improvement}")
    
    # Early stopping
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n⚠️  Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
        break

# Load best model
model.load_state_dict(torch.load("best_model.pth"))
print("\n✓ Loaded best model from training\n")

# Plot training progress
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_sharpes, label='Train Sharpe', linewidth=2)
plt.plot(val_sharpes, label='Val Sharpe', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Sharpe Ratio')
plt.title('Training Progress: Sharpe Ratio')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
epochs_range = range(1, len(train_sharpes) + 1)
plt.plot(epochs_range, train_sharpes, label='Train', linewidth=2)
plt.plot(epochs_range, val_sharpes, label='Validation', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Sharpe')
plt.xlabel('Epoch')
plt.ylabel('Sharpe Ratio')
plt.title('Sharpe Ratio Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_progress.png", dpi=150)
print("Saved 'training_progress.png'")

# ---------- STEP 6: Evaluation ----------
model.eval()
with torch.no_grad():
    X_test_t = torch.from_numpy(X_test).float().to(DEVICE)
    w_test = model(X_test_t).cpu().numpy()

pv_nn = [1.0]
pv_eq = [1.0]
eq_weights = np.ones(num_assets) / num_assets

for i in range(len(Y_test)):
    r = Y_test[i]
    pv_nn.append(pv_nn[-1] * (1 + np.dot(w_test[i], r)))
    pv_eq.append(pv_eq[-1] * (1 + np.dot(eq_weights, r)))

plt.figure(figsize=(14,7))
plt.plot(pv_nn, label="AI Portfolio (with Sectors)", linewidth=2, color='#2E86AB')
plt.plot(pv_eq, label="Equal Weight Baseline", linestyle="--", linewidth=2, color='#A23B72')
plt.title(f"Portfolio Performance on Test Data ({len(Y_test)} days)", fontsize=14, fontweight='bold')
plt.xlabel("Days", fontsize=12)
plt.ylabel("Portfolio Value", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("performance_chart.png", dpi=150)
print("Saved 'performance_chart.png'")

m_nn = portfolio_metrics(np.array(pv_nn))
m_eq = portfolio_metrics(np.array(pv_eq))
print("\n" + "="*60)
print("BACKTEST RESULTS")
print("="*60)
print(f"AI Portfolio:")
print(f"  Cumulative Return: {m_nn['cum_ret']*100:>8.2f}%")
print(f"  Annual Return:     {m_nn['ann_ret']*100:>8.2f}%")
print(f"  Annual Volatility: {m_nn['ann_vol']*100:>8.2f}%")
print(f"  Sharpe Ratio:      {m_nn['sharpe']:>8.2f}")
print(f"  Max Drawdown:      {m_nn['max_dd']*100:>8.2f}%")
print(f"\nEqual Weight Baseline:")
print(f"  Cumulative Return: {m_eq['cum_ret']*100:>8.2f}%")
print(f"  Annual Return:     {m_eq['ann_ret']*100:>8.2f}%")
print(f"  Annual Volatility: {m_eq['ann_vol']*100:>8.2f}%")
print(f"  Sharpe Ratio:      {m_eq['sharpe']:>8.2f}")
print(f"  Max Drawdown:      {m_eq['max_dd']*100:>8.2f}%")

# ---------- STEP 7: TOP 10 PREDICTIONS ----------
print("\n" + "="*60)
print("GENERATING NEXT-DAY ALLOCATION")
print("="*60)

c_last = close_mat.iloc[-WINDOW:].to_numpy()
h_last = high_mat.iloc[-WINDOW:].to_numpy()
l_last = low_mat.iloc[-WINDOW:].to_numpy()

final_close = c_last[-1, :]
c_norm_last = c_last / final_close
h_norm_last = h_last / final_close
l_norm_last = l_last / final_close

price_features_last = np.stack([l_norm_last, h_norm_last, c_norm_last], axis=0)
sector_features_last = sector_encoding.T[:, np.newaxis, :]
sector_features_last = np.repeat(sector_features_last, WINDOW, axis=1)

X_final = np.concatenate([price_features_last, sector_features_last], axis=0)
X_final_t = torch.from_numpy(X_final).float().unsqueeze(0).to(DEVICE)

with torch.no_grad():
    future_weights = model(X_final_t).cpu().numpy().flatten()

# Get top K allocations
top_k_indices = np.argsort(future_weights)[-TOP_K:][::-1]

alloc_data = []
for idx in top_k_indices:
    ticker = TICKERS[idx]
    weight = future_weights[idx] * 100
    sector = sector_of(ticker)
    alloc_data.append({
        "Rank": len(alloc_data) + 1,
        "Ticker": ticker,
        "Sector": sector,
        "Allocation": weight
    })

alloc_df = pd.DataFrame(alloc_data)

print(f"\nTop {TOP_K} Recommended Positions:")
print(alloc_df.to_string(index=False))

# Show sector distribution
sector_alloc = alloc_df.groupby('Sector')['Allocation'].sum().sort_values(ascending=False)
print("\nSector Distribution:")
for sector, alloc in sector_alloc.items():
    print(f"  {sector:15s}: {alloc:6.2f}%")

alloc_df.to_csv("final_allocation.csv", index=False)
print(f"\nSaved recommendations to 'final_allocation.csv'")
print("="*60)