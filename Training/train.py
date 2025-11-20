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
CSV_PATH = "stocks_market_data.csv"   # make sure this exists in working dir
TICKERS = ["AAPL", "MSFT", "GOOGL"]   # adjust as needed (must match CSV)
WINDOW = 50                           # number of timesteps per sample (paper used 50)
FEATURES = ["Low", "High", "Close"]   # order matters (we'll stack Low, High, Close)
TRAIN_TEST_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- UTIL: metrics ----------
def portfolio_metrics(portfolio_values, periods_per_year=252):
    # portfolio_values: 1D array of daily (or period) cumulative values (starting at 1.0)
    rets = np.diff(portfolio_values) / portfolio_values[:-1]
    if len(rets) == 0:
        return {}
    cumulative_return = portfolio_values[-1] - 1.0
    total_periods = len(rets)
    annual_factor = periods_per_year / total_periods * len(rets)
    # actually compute annualized return:
    days = total_periods
    ann_return = (portfolio_values[-1]) ** (periods_per_year / days) - 1
    ann_vol = np.std(rets) * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    # max drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdowns = (running_max - portfolio_values) / running_max
    max_dd = drawdowns.max()
    return {"cumulative_return": cumulative_return,
            "annual_return": ann_return,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd}

# ---------- STEP 1: load and pivot CSV ----------
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Ensure Date column exists and is datetime
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])
else:
    # try common index name variants
    raise RuntimeError("CSV missing 'Date' column. Reset index in CSV creation step.")

# Filter tickers present in CSV
present_tickers = df["Ticker"].unique().tolist()
missing = [t for t in TICKERS if t not in present_tickers]
if missing:
    raise RuntimeError(f"Tickers missing in CSV: {missing}. CSV has: {present_tickers}")

# Pivot each feature to matrix: index=Date, columns=Ticker
def pivot_feature(df, feature):
    p = df.pivot(index="Date", columns="Ticker", values=feature)
    p = p.sort_index()
    # keep only user tickers and keep column order stable
    p = p[TICKERS]
    return p

close_mat = pivot_feature(df, "Close")
high_mat  = pivot_feature(df, "High")
low_mat   = pivot_feature(df, "Low")

# Align dates across all features (inner join)
common_dates = close_mat.index.intersection(high_mat.index).intersection(low_mat.index)
close_mat = close_mat.loc[common_dates]
high_mat  = high_mat.loc[common_dates]
low_mat   = low_mat.loc[common_dates]

print("Data range:", common_dates.min(), "to", common_dates.max())
print("Number of aligned dates:", len(common_dates))

# ---------- STEP 2: fill missing -> simple but sensible ----------
# Forward-fill then backward-fill per asset. If still NaN, drop those dates.
close_mat = close_mat.fillna(method="ffill").fillna(method="bfill")
high_mat  = high_mat.fillna(method="ffill").fillna(method="bfill")
low_mat   = low_mat.fillna(method="ffill").fillna(method="bfill")

# Drop any remaining rows with NaN
mask = close_mat.isnull().any(axis=1) | high_mat.isnull().any(axis=1) | low_mat.isnull().any(axis=1)
if mask.any():
    print("Dropping rows with remaining NaNs:", mask.sum())
    close_mat = close_mat[~mask]
    high_mat = high_mat[~mask]
    low_mat = low_mat[~mask]

# ---------- STEP 3: build tensors and next-period returns ----------
dates = close_mat.index.to_numpy()
num_dates = len(dates)
num_assets = len(TICKERS)

X_list = []
Y_list = []        # next-period simple returns: (close_next / close_last) - 1
y_dates = []

for i in range(WINDOW, num_dates - 1):  # -1 because we need t+1 return
    sl = slice(i - WINDOW, i)
    c = close_mat.iloc[sl].to_numpy()   # shape (WINDOW, num_assets)
    h = high_mat.iloc[sl].to_numpy()
    l = low_mat.iloc[sl].to_numpy()

    last_close = c[-1, :]                # shape (num_assets,)
    # avoid division by zero
    last_close[last_close == 0] = 1.0

    # normalize (paper-style): divide each time series by last close -> relative movements
    c_norm = c / last_close[np.newaxis, :]
    h_norm = h / last_close[np.newaxis, :]
    l_norm = l / last_close[np.newaxis, :]

    # stack features (features, window, assets)
    X = np.stack([l_norm.T, h_norm.T, c_norm.T], axis=0)  # careful: we want (3, assets, window) or (3, window, assets)
    # We'll standardize to (features, window, assets) like paper: (3, WINDOW, num_assets)
    # Currently l_norm.shape = (WINDOW, num_assets) -> l_norm.T is (num_assets, WINDOW)
    # So we need final shape (3, WINDOW, num_assets):
    X = np.stack([l_norm, h_norm, c_norm], axis=0)  # (3, WINDOW, num_assets)

    # next-period return using close at i (last_close) and close at i+1
    close_next = close_mat.iloc[i + 0].to_numpy()  # careful on indexing: we used windows to i-1? we used slice up to i-1? we used sl = [i-window, i)
    # Since sl ends at i-1, last_close corresponds to row at i-1. So next close should be at i
    close_next = close_mat.iloc[i].to_numpy()
    r_next = (close_next / last_close) - 1.0   # simple return vector
    X_list.append(X)      # X shape: (3, WINDOW, num_assets)
    Y_list.append(r_next)
    y_dates.append(dates[i])

X_array = np.stack(X_list, axis=0)   # (samples, 3, WINDOW, num_assets)
Y_array = np.stack(Y_list, axis=0)   # (samples, num_assets)
y_dates = np.array(y_dates)

print("Built X_array shape:", X_array.shape)
print("Built Y_array shape:", Y_array.shape)

# Save arrays for quick reuse
np.savez_compressed("tensors.npz", X=X_array, Y=Y_array, dates=y_dates, tickers=np.array(TICKERS))
print("Saved tensors.npz")

# ---------- STEP 4: train/test split ----------
X_train, X_test, Y_train, Y_test = train_test_split(X_array, Y_array, test_size=TRAIN_TEST_SPLIT, random_state=SEED, shuffle=False)
print("Train samples:", X_train.shape[0], "Test samples:", X_test.shape[0])

# ---------- STEP 5: Baselines ----------
def equal_weight_alloc(num_assets):
    return np.ones(num_assets) / num_assets

def inv_vol_alloc(returns_matrix):
    # returns_matrix shape: (samples, num_assets) where returns are simple returns per period
    vol = np.std(returns_matrix, axis=0, ddof=1)
    # avoid zero vol
    vol[vol == 0] = 1e-8
    w = 1.0 / vol
    w = w / w.sum()
    return w

# compute a baseline allocation from training set next returns
baseline_inv_vol = inv_vol_alloc(Y_train)
print("Inverse-vol baseline weights:", baseline_inv_vol)

# Evaluate baseline strategies across test period: apply static weights each period and compute cumulative portfolio value
def simulate_static_weights(weights, start_index=0, Y_series=Y_test):
    # Y_series: shape (samples, assets) with per-period simple returns (next period)
    pv = [1.0]
    for r in Y_series:
        # portfolio return = w dot r
        pr = np.dot(weights, r)
        pv.append(pv[-1] * (1.0 + pr))
    return np.array(pv)

# test static baselines
ew_pv = simulate_static_weights(equal_weight_alloc(num_assets), Y_series=Y_test)
invvol_pv = simulate_static_weights(baseline_inv_vol, Y_series=Y_test)

print("Equal-weight test final:", ew_pv[-1])
print("Inv-vol test final:", invvol_pv[-1])

print("EW metrics:", portfolio_metrics(ew_pv))
print("Inv-vol metrics:", portfolio_metrics(invvol_pv))

# ---------- STEP 6: Simple neural-network allocator (predicts weights directly) ----------
# We'll build a small model: input -> conv2d -> flatten -> dense -> weights (softmax)
class AllocNet(nn.Module):
    def __init__(self, num_assets, window, features=3, hidden=256):
        super().__init__()
        # input shape (batch, features, window, assets)
        # we'll use a small Conv2d
        self.conv = nn.Conv2d(in_channels=features, out_channels=16, kernel_size=(3,3), padding=1)
        # compute flatten size roughly: 16 * window * num_assets
        flat_size = 16 * window * num_assets
        self.fc1 = nn.Linear(flat_size, hidden)
        self.fc2 = nn.Linear(hidden, num_assets)
    def forward(self, x):
        # x: (batch, features, window, assets)
        z = torch.relu(self.conv(x))
        z = z.view(z.size(0), -1)
        z = torch.relu(self.fc1(z))
        w_raw = self.fc2(z)
        w = torch.softmax(w_raw, dim=1)
        return w

# convert data to torch tensors
def to_torch_np(X_np, Y_np):
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    # reorder to (batch, features, window, assets) already in that shape
    Y_t = torch.tensor(Y_np, dtype=torch.float32).to(DEVICE)
    return X_t, Y_t

model = AllocNet(num_assets=num_assets, window=WINDOW, features=3, hidden=256).to(DEVICE)
opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)

# custom loss: negative expected portfolio return over batch (+ small entropy regularization to encourage diversification)
def loss_fn(weights, next_returns, entropy_coef=1e-3):
    # weights: (batch, num_assets)
    # next_returns: (batch, num_assets) simple returns for next period
    # portfolio return per sample:
    port_rets = torch.sum(weights * next_returns, dim=1)  # (batch,)
    # we want to maximize mean(port_rets) -> minimize negative mean
    loss_return = -torch.mean(port_rets)
    # entropy regularization (maximize entropy => minimize negative entropy)
    eps = 1e-8
    entropy = -torch.sum(weights * torch.log(weights + eps), dim=1).mean()
    loss = loss_return - entropy_coef * entropy
    return loss, loss_return.item(), entropy.item()

# Training loop (mini-batch)
X_train_t, Y_train_t = to_torch_np(X_train, Y_train)
X_test_t, Y_test_t = to_torch_np(X_test, Y_test)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_t, Y_train_t),
                                           batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        opt.zero_grad()
        w = model(xb)            # (batch, assets)
        loss, _, _ = loss_fn(w, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / X_train_t.size(0)
    if epoch % 10 == 0 or epoch == 1:
        # evaluate
        model.eval()
        with torch.no_grad():
            w_test = model(X_test_t).cpu().numpy()      # (test_samples, assets)
            # simulate trading: for every test sample, use weights predicted at time t to invest for period t (uses Y_test)
            pv = [1.0]
            for i in range(len(w_test)):
                pr = np.dot(w_test[i], Y_test[i])
                pv.append(pv[-1] * (1.0 + pr))
            pv = np.array(pv)
            metrics = portfolio_metrics(pv)
        print(f"Epoch {epoch:03d} avg_loss={avg_loss:.6f} test_final={pv[-1]:.4f} sharpe={metrics.get('sharpe'):.4f}")

# Save model
torch.save(model.state_dict(), "alloc_net.pth")
print("Saved model to alloc_net.pth")

# ---------- STEP 7: Full evaluation: compare NN vs baselines ----------
model.eval()
with torch.no_grad():
    w_test = model(X_test_t).cpu().numpy()
    pv_nn = [1.0]
    pv_ew = [1.0]
    pv_inv = [1.0]
    for i in range(len(w_test)):
        r = Y_test[i]
        pv_nn.append(pv_nn[-1] * (1.0 + np.dot(w_test[i], r)))
        pv_ew.append(pv_ew[-1] * (1.0 + np.dot(equal_weight_alloc(num_assets), r)))
        pv_inv.append(pv_inv[-1] * (1.0 + np.dot(baseline_inv_vol, r)))

pv_nn = np.array(pv_nn)
pv_ew = np.array(pv_ew)
pv_inv = np.array(pv_inv)

print("NN final:", pv_nn[-1], portfolio_metrics(pv_nn))
print("EW final:", pv_ew[-1], portfolio_metrics(pv_ew))
print("InvVol final:", pv_inv[-1], portfolio_metrics(pv_inv))

# Plot cumulative returns
plt.figure(figsize=(10,6))
plt.plot(pv_nn, label="NN allocator")
plt.plot(pv_ew, label="Equal Weight")
plt.plot(pv_inv, label="Inv Vol")
plt.legend()
plt.xlabel("Test period step")
plt.ylabel("Portfolio value (start=1.0)")
plt.title("Test set cumulative wealth")
plt.grid(True)
plt.savefig("performance.png")
print("Saved performance.png")

print("Done.")

# ---------- FINAL EXPECTED OUTPUT: Allocation Table ----------

# get last sample in test set as current state
X_latest = X_test_t[-1].unsqueeze(0)  # shape (1, features, window, assets)

model.eval()
with torch.no_grad():
    final_weights = model(X_latest).cpu().numpy().flatten()  # shape (num_assets,)

# Format as table
alloc_df = pd.DataFrame({
    "Ticker": TICKERS,
    "Allocation %": (final_weights * 100).round(2)
})

print("\n===== FINAL RECOMMENDED PORTFOLIO ALLOCATION =====\n")
print(alloc_df)

# Save to CSV
alloc_df.to_csv("final_portfolio_allocation.csv", index=False)
print("\nSaved final_portfolio_allocation.csv")
