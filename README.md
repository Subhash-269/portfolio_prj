# AI-Powered Portfolio Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Django](https://img.shields.io/badge/Django-5.0-green) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)

## Context
An investment portfolio is an accumulation of different investable assets owned by an individual or institution. Creating a personalized investment portfolio requires balancing goals, risk tolerance, and time horizons (Campbell & Safane, 2024).

**Portfolio Optimization** is the mathematical process of selecting the best combination of assets to achieve the highest possible return for a given level of risk. Traditionally, this is based on **Modern Portfolio Theory (MPT)**, developed by Nobel laureate Harry Markowitz (Victoria, 2024).

## Problem Statement
**Why isn't Traditional MPT enough?**
While MPT is foundational, it relies on assumptions that often fail in real-world markets:
* **Linear Assumptions:** It assumes linear correlations between assets, whereas markets are chaotic and non-linear.
* **Normal Distribution:** It assumes returns follow a bell curve, ignoring "fat tails" and Black Swan events (market crashes).
* **Static Nature:** Traditional formulas struggle to adapt quickly to changing market regimes (e.g., shifting from a bull market to high volatility).

**What does this AI approach bring to the table?**
This project moves beyond static formulas by utilizing **Deep Learning (Neural Networks)** to optimize stock allocation.
* **Pattern Recognition:** The Neural Network identifies complex, non-linear patterns in historical price data that standard variance formulas miss.
* **Dynamic Adaptation:** The model adjusts weights based on sliding windows of data, reacting to recent trends rather than long-term historical averages.
* **Direct Optimization:** Instead of forecasting prices (which is error-prone), our model optimizes the *allocation weights* directly to maximize the Sharpe Ratio.

## Solution
<!-- We have built a **Django-based API** that utilizes a PyTorch Neural Network to generate optimal portfolio allocations.

* **Algorithm:** Convolutional Neural Network (CNN) specifically tuned for time-series financial data.
* **Objective Function:** A custom loss function that maximizes the **Sharpe Ratio** (Risk-Adjusted Return) while maintaining portfolio diversity via Entropy regularization.
* **Delivery:** A REST API (documented with Swagger) that accepts a list of tickers and returns the optimal percentage allocation. -->

An AI investment engine that uses deep learning to maximize risk-adjusted returns through dynamic, non-linear asset allocation.
<!-- 
### Expected Outcome
* **Input:** A list of stock tickers (e.g., `['AAPL', 'MSFT', 'GOOG']`).
* **Output:** Precise allocation percentages for the next trading period.
    ```json
    {
      "AAPL": 33.12,
      "MSFT": 35.17,
      "GOOG": 31.71
    }
    ```

## Methodology
The system follows a sliding-window approach to training:

1.  **Data Ingestion:** Load historical OHLC (Open, High, Low, Close) data.
2.  **Preprocessing:** Normalize price data relative to the start of the window.
3.  **Feature Engineering:** Create tensor windows of shape `(3, Window_Size, Num_Assets)`.
4.  **Training:** The `AllocNet` model processes these windows to output a Softmax probability distribution (weights summing to 1.0).
5.  **Validation:** The model is evaluated against an Equal-Weight baseline using Cumulative Return and Sharpe Ratio metrics.

## Existing Approaches & Related Work
This project draws inspiration from several key open-source financial AI projects:

* **[PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio)** (Zhengyao Jiang et al.)
    * *Concept:* A Deep Reinforcement Learning Framework for Financial Portfolio Management.
    * *Relevance:* Utilizes policy-gradient approaches tailored for portfolio management.
* **[DeepDow](https://github.com/jankrepl/deepdow)**
    * *Concept:* End-to-end differentiable portfolio optimization.
    * *Relevance:* Connects deep learning features directly to the allocation layer, similar to our neural network approach.
* **[RLPortfolio](https://github.com/CaioSBC/RLPortfolio)**
    * *Concept:* RL agents + environment for portfolio optimization.
    * *Relevance:* Provides strong baselines for simulation environments.

## Installation & Usage -->

### Prerequisites
* Python 3.8+
* `pip`

### 1. Clone and Install
```bash
git clone [https://github.com/Subhash-269/portfolio_prj.git](https://github.com/Subhash-269/portfolio_prj.git)
cd portfolio_prj
pip install -r requirements.txt
