# pipeline.py
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
# ---------- CONFIG ----------
# Use the new file name from the previous step
CSV_PATH = "diversified_market_data.csv" 
# If that file doesn't exist, fall back to the old one
if not os.path.exists(CSV_PATH):
    CSV_PATH = "stocks_market_data.csv"

WINDOW = 50                       # timesteps per sample
TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 20                   # Increased slightly for better convergence
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Set seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)

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

# ---------- STEP 1: Load and Pivot CSV ----------
print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# 1. Clean Dates
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
else:
    raise RuntimeError("CSV missing 'Date' column.")

# 2. Auto-detect Tickers (Dynamic)
TICKERS = df["Ticker"].unique().tolist()
TICKERS.sort() # Sort alphabetically to keep consistent order
print(f"Detected {len(TICKERS)} tickers: {TICKERS}")

# 3. Pivot features
def pivot_feature(df, feature):
    p = df.pivot(index="Date", columns="Ticker", values=feature)
    p = p.sort_index()
    # Ensure columns match our sorted TICKERS list
    p = p[TICKERS] 
    return p

try:
    close_mat = pivot_feature(df, "Close")
    high_mat  = pivot_feature(df, "High")
    low_mat   = pivot_feature(df, "Low")
except KeyError as e:
    print(f"Error: Column {e} not found in CSV. Check your data headers.")
    exit()

# 4. Align dates
common_dates = close_mat.index.intersection(high_mat.index).intersection(low_mat.index)
close_mat = close_mat.loc[common_dates]
high_mat  = high_mat.loc[common_dates]
low_mat   = low_mat.loc[common_dates]

# ---------- STEP 2: Fill Missing Data ----------
# Use modern pandas syntax (ffill/bfill)
close_mat = close_mat.ffill().bfill()
high_mat  = high_mat.ffill().bfill()
low_mat   = low_mat.ffill().bfill()

# Drop dates that are still incomplete
mask = close_mat.isnull().any(axis=1) | high_mat.isnull().any(axis=1) | low_mat.isnull().any(axis=1)
if mask.any():
    print(f"Dropping {mask.sum()} rows with NaNs.")
    close_mat = close_mat[~mask]
    high_mat = high_mat[~mask]
    low_mat = low_mat[~mask]

if len(close_mat) < WINDOW + 1:
    raise RuntimeError(f"Not enough data! Need > {WINDOW} rows, but got {len(close_mat)}.")

# ---------- STEP 3: Build Tensors ----------
dates = close_mat.index.to_numpy()
num_dates = len(dates)
num_assets = len(TICKERS)

X_list = []
Y_list = []       
y_dates = []

# Create sliding windows
for i in range(WINDOW, num_dates - 1):
    # Inputs: t-WINDOW to t-1
    sl = slice(i - WINDOW, i)
    
    c = close_mat.iloc[sl].to_numpy()
    h = high_mat.iloc[sl].to_numpy()
    l = low_mat.iloc[sl].to_numpy()

    # Normalize by the LAST close in the window
    last_close = c[-1, :]
    last_close[last_close == 0] = 1.0 # safety

    c_norm = c / last_close
    h_norm = h / last_close
    l_norm = l / last_close

    # Shape: (3 features, WINDOW, assets)
    X = np.stack([l_norm, h_norm, c_norm], axis=0)

    # Target: Return from t to t+1
    p_current = close_mat.iloc[i].to_numpy()
    r_next = (p_current / last_close) - 1.0
    
    X_list.append(X)
    Y_list.append(r_next)
    y_dates.append(dates[i])

X_array = np.stack(X_list, axis=0) # (samples, 3, WINDOW, assets)
Y_array = np.stack(Y_list, axis=0) # (samples, assets)

print(f"Dataset created: {X_array.shape} samples")

# ---------- STEP 4: Train/Test Split ----------
X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=TRAIN_TEST_SPLIT, shuffle=False)

# ---------- STEP 5: Neural Net ----------

# hyperparams for memory-friendly run
BATCH_SIZE = 16   # smaller batches use less memory
CONV_CHANNELS = 8 # reduce from 32 -> 8
HIDDEN = 64       # reduce hidden size

# re-define a smaller model matching new hyperparams
class AllocNetSmall(nn.Module):
    def __init__(self, num_assets, window, features=3, conv_ch=CONV_CHANNELS, hidden=HIDDEN):
        super().__init__()
        self.conv = nn.Conv2d(features, conv_ch, kernel_size=(3,1)) 
        conv_out_h = window - 2
        flat_size = conv_ch * conv_out_h * num_assets
        # To avoid extremely large fc, project by a 1x1 conv over assets to shrink dimension first
        # (optional: comment this if not desired)
        # we'll insert a 1x1 conv to reduce asset dimension: Conv2d(in_channels=conv_ch, out_channels=conv_ch, kernel=(1,1))
        self.shrink = nn.Conv2d(conv_ch, conv_ch, kernel_size=(1,1))
        self.fc1 = nn.Linear(flat_size, hidden)
        self.fc2 = nn.Linear(hidden, num_assets)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, features, window, assets)
        z = torch.relu(self.conv(x))           # (batch, conv_ch, window-2, assets)
        z = torch.relu(self.shrink(z))         # small 1x1 projection
        z = z.view(z.size(0), -1)
        z = self.dropout(torch.relu(self.fc1(z)))
        w_raw = self.fc2(z)
        w = torch.softmax(w_raw, dim=1)
        return w


# Create Dataset class that keeps arrays on CPU and yields batches
class TensorDatasetOnCPU(data.Dataset):
    def __init__(self, X_np, Y_np):
        self.X = X_np.astype(np.float32)  # keep as float32 on CPU
        self.Y = Y_np.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
# Updated Loss Function: The "Goldilocks" Tuning
# Replace the existing loss_fn with this:
# REPLACEMENT LOSS FUNCTION
def loss_fn(weights, next_returns):
    # Calculate portfolio returns for the batch
    port_rets = torch.sum(weights * next_returns, dim=1)
    
    # 1. Compute Sharpe Ratio (Reward / Risk)
    # We add a tiny number (1e-6) to prevent division by zero
    mean_ret = torch.mean(port_rets)
    vol_ret = torch.std(port_rets) + 1e-6
    sharpe = mean_ret / vol_ret

    # 2. Gentle Entropy (Diversity)
    # We LOWERED this coefficient from 0.05 to 0.001
    # This allows the model to have "opinions" (uneven weights)
    entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=1).mean()
    
    # LOSS: We want to MAXIMIZE Sharpe. So we MINIMIZE negative Sharpe.
    loss = -(sharpe + 0.001 * entropy) 
    return loss
class LossFunctions:
    @staticmethod
    def explicit_log_return(weights, price_change_vector, prev_weights, commission=0.0025, epsilon=1e-8):
        """
        Robustized explicit log-return loss.
        price_change_vector: can be either (1 + simple_returns) OR raw price-change factors.
                              We will detect and convert if necessary.
        weights, prev_weights: torch tensors, shape (batch, assets)
        """
        # ensure tensors
        # price_change_vector expected > 0 (factors). If values look like small around zero (e.g. -0.01..0.01),
        # convert to factors:
        if price_change_vector.min() < 0.5:  # heuristic: if there exist values < 0.5 it's probably simple returns
            y_factors = price_change_vector + 1.0
        else:
            y_factors = price_change_vector

        # step_return = w_t dot y_factors (gives factor of portfolio growth per sample)
        # But paper uses mu_t and w_{t-1}, we'll use standard approximate approach:
        step_factor = torch.sum(weights * y_factors, dim=1)  # (batch,)
        # turnover cost
        turnover = torch.sum(torch.abs(weights - prev_weights), dim=1)  # (batch,)
        transaction_cost = commission * turnover

        # net factor after cost: step_factor - cost (this must remain > 0)
        # but subtracting cost from factor is not ideal (factor ~1.x, cost ~0.x). 
        # Better: multiply factor by (1 - transaction_cost) OR subtract small amount
        net_factor = step_factor * (1.0 - transaction_cost)
        # clamp to avoid <= 0
        net_factor_clamped = torch.clamp(net_factor, min=epsilon)

        # Use log of net factor as reward; loss = negative mean log factor
        log_return = torch.log(net_factor_clamped)
        loss = -torch.mean(log_return)

        # safe diagnostic: if loss is NaN, return a large positive value to avoid breaking training
        if torch.isnan(loss):
            # return a tensor of big loss so optimizer moves away
            return torch.tensor(1e6, device=weights.device, dtype=weights.dtype)
        return loss


train_dataset = TensorDatasetOnCPU(X_train, Y_train)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

# instantiate smaller model
model = AllocNetSmall(num_assets=num_assets, window=WINDOW).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)

# convenience: function to move numpy batch to device
def batch_to_device(batch):
    xb_np, yb_np = batch
    xb = torch.tensor(xb_np, dtype=torch.float32).to(DEVICE)  # (B, 3, W, A)
    yb = torch.tensor(yb_np, dtype=torch.float32).to(DEVICE)
    return xb, yb

print("Batched training start. Using device:", DEVICE)
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    count = 0
    for batch in train_loader:
        xb, yb = batch_to_device(batch)   # xb, yb are tensors on DEVICE
        opt.zero_grad()
        w = model(xb)                     # (batch, assets)
        yb_factors = 1.0 + yb
        # use prev_weights = w.detach() as a simple approximation
        loss = LossFunctions.explicit_log_return(w, yb_factors, w.detach(), commission=0.0025)
        # detect NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: NaN/Inf loss encountered; skipping batch")
            continue
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        epoch_loss += loss.item() * xb.size(0)
        count += xb.size(0)
        # free GPU cache occasionally
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    avg_loss = epoch_loss / count
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} avg_loss={avg_loss:.6f}")
# Save model
torch.save(model.state_dict(), "alloc_net_small.pth")

# ---------- STEP 6: Evaluation ----------
# Minimal proper evaluation (convert numpy -> tensor and move to DEVICE)
model.eval()
with torch.no_grad():
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)   # convert + move
    w_test = model(X_test_t).cpu().numpy()                           # forward + move back to CPU


# Calculate cumulative returns
pv_nn = [1.0]
pv_eq = [1.0] # Equal weight baseline
eq_weights = np.ones(num_assets) / num_assets

for i in range(len(Y_test)):
    r = Y_test[i]
    pv_nn.append(pv_nn[-1] * (1 + np.dot(w_test[i], r)))
    pv_eq.append(pv_eq[-1] * (1 + np.dot(eq_weights, r)))

# Plot
plt.figure(figsize=(12,6))
plt.plot(pv_nn, label="AI Agent Portfolio")
plt.plot(pv_eq, label="Equal Weight Baseline", linestyle="--")
plt.title(f"Portfolio Performance on Test Data ({len(Y_test)} days)")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True)
plt.savefig("performance_chart.png")
print("Saved 'performance_chart.png'")

# Metrics
m_nn = portfolio_metrics(np.array(pv_nn))
m_eq = portfolio_metrics(np.array(pv_eq))
print("\n--- Results ---")
print(f"AI Agent: Sharpe={m_nn['sharpe']:.2f}, Return={m_nn['cum_ret']*100:.1f}%")
print(f"Baseline: Sharpe={m_eq['sharpe']:.2f}, Return={m_eq['cum_ret']*100:.1f}%")

# ---------- STEP 7: PREDICT FOR TOMORROW ----------
# We take the absolute last 'WINDOW' days from the FULL dataset to predict allocation for the future
print("\nGenerating allocation for the next trading day...")

c_last = close_mat.iloc[-WINDOW:].to_numpy()
h_last = high_mat.iloc[-WINDOW:].to_numpy()
l_last = low_mat.iloc[-WINDOW:].to_numpy()

# Normalize by the very last close
final_close = c_last[-1, :]
c_norm_last = c_last / final_close
h_norm_last = h_last / final_close
l_norm_last = l_last / final_close

# Build tensor (1 sample, 3 features, WINDOW, assets)
X_final = np.stack([l_norm_last, h_norm_last, c_norm_last], axis=0)
X_final_t = torch.tensor(X_final, dtype=torch.float32).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    future_weights = model(X_final_t).cpu().numpy().flatten()


# Filter for significant allocations (> 1%)
alloc_data = []
for i, ticker in enumerate(TICKERS):
    pct = future_weights[i] * 100
    if pct > 1.0: # Only show if allocation is > 1%
        alloc_data.append({"Ticker": ticker, "Allocation": pct})

alloc_df = pd.DataFrame(alloc_data).sort_values("Allocation", ascending=False)

print("\n===== RECOMMENDED PORTFOLIO =====")
print(alloc_df.to_string(index=False, float_format="%.2f%%"))
alloc_df.to_csv("final_allocation.csv", index=False, float_format="%.2f")