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

# ---------- CONFIG ----------
# Use the new file name from the previous step
CSV_PATH = "diversified_market_data.csv" 
# If that file doesn't exist, fall back to the old one
if not os.path.exists(CSV_PATH):
    CSV_PATH = "stocks_market_data.csv"

WINDOW = 50                       # timesteps per sample
TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 100                      # Increased slightly for better convergence
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
class AllocNet(nn.Module):
    def __init__(self, num_assets, window, features=3, hidden=128):
        super().__init__()
        # Conv layer to extract temporal features per asset
        self.conv = nn.Conv2d(features, 32, kernel_size=(3,1)) 
        # Flatten size depends on window size after conv
        # window - 3 + 1 = window - 2
        conv_out_h = window - 2
        flat_size = 32 * conv_out_h * num_assets
        
        self.fc1 = nn.Linear(flat_size, hidden)
        self.fc2 = nn.Linear(hidden, num_assets)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, features, window, assets)
        z = torch.relu(self.conv(x)) 
        z = z.view(z.size(0), -1)
        z = self.dropout(torch.relu(self.fc1(z)))
        w_raw = self.fc2(z)
        w = torch.softmax(w_raw, dim=1) # Ensure weights sum to 1
        return w

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
    def explicit_log_return(weights, price_change_vector, prev_weights, commission=0.0025):
        """
        Implementation of Eq. 21 from the paper "Deep Portfolio Management".
        
        weights: (batch, num_assets) -> The output of your neural net (w_t)
        price_change_vector: (batch, num_assets) -> y_t = price_t / price_{t-1}
        prev_weights: (batch, num_assets) -> w_{t-1} (from memory)
        commission: transaction fee rate (e.g., 0.25%)
        """
        
        # 1. Calculate the portfolio value vector BEFORE transaction costs
        # dot product of weights and price changes
        # (batch_size, )
        step_return = torch.sum(weights * price_change_vector, dim=1)
        
        # 2. Calculate Transaction Cost Factor (mu_t)
        # The paper uses a recursive formula, but we can approximate it for stability:
        # Cost = commission * sum(|w_t - w_{t-1}|)
        # This is the cost of rebalancing.
        turnover = torch.sum(torch.abs(weights - prev_weights), dim=1)
        transaction_cost = commission * turnover
        
        # 3. Calculate Net Return
        # The paper defines r_t = ln(mu_t * y_t * w_{t-1})
        # Which simplifies roughly to: ln(step_return - transaction_cost)
        
        # Add epsilon to prevent log(0) or log(negative)
        net_step_return = step_return - transaction_cost + 1e-8
        
        # 4. Logarithmic Return (The "Kelly" Logic)
        log_return = torch.log(net_step_return)
        
        # 5. Loss = Negative Average Log Return
        loss = -torch.mean(log_return)
        
        return loss

# Setup Model
model = AllocNet(num_assets, WINDOW).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR)

# Training Loop
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(DEVICE)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

print("Training...")
for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()
    w = model(X_train_t)
    loss = LossFunctions.explicit_log_return(w, Y_train_t, w)  # Assuming prev_weights = current weights for simplicity
    loss.backward()
    opt.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.6f}")

# ---------- STEP 6: Evaluation ----------
model.eval()
with torch.no_grad():
    w_test = model(X_test_t).cpu().numpy()

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