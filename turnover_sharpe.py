
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(0)
T = 1000    
price_x = np.exp(np.cumsum(np.random.normal(0, 0.01, T)))
price_y = np.exp(np.cumsum(np.random.normal(0, 0.01, T)))

x = np.log(pd.Series(price_x))
y = np.log(pd.Series(price_y))

X = sm.add_constant(x.values)
beta_static = sm.OLS(y.values, X).fit().params[1]
beta_static_series = np.full(T, beta_static)

def kalman_beta(x, y, R, Q=1e-4):
    T = len(x)
    beta = np.zeros(T)
    P = np.zeros(T)

    beta[0] = 0.0
    P[0] = 1.0

    for t in range(1, T):
        beta_pred = beta[t-1]
        P_pred = P[t-1] + R

        H = x.iloc[t]
        S = H**2 * P_pred + Q
        K = P_pred * H / S

        beta[t] = beta_pred + K * (y.iloc[t] - H * beta_pred)
        P[t] = (1 - K * H) * P_pred

    return beta

def trading_returns(x, y, beta):
    spread = y - beta * x
    z = (spread - spread.mean()) / spread.std()

    position = np.sign(z)         
    pnl = position.shift(1) * (-spread.diff())
    pnl = pnl.dropna()

    return pnl

def sharpe(pnl):
    return np.sqrt(252) * pnl.mean() / pnl.std()

def turnover(beta):
    return np.mean(np.abs(np.diff(beta)))

R_grid = np.logspace(-6, -2, 15)

turnovers = []
sharpes = []

for R in R_grid:
    beta_kf = kalman_beta(x, y, R)
    pnl = trading_returns(x, y, beta_kf)

    turnovers.append(turnover(beta_kf))
    sharpes.append(sharpe(pnl))

pnl_static = trading_returns(x, y, beta_static_series)
turnover_static = turnover(beta_static_series)
sharpe_static = sharpe(pnl_static)

plt.figure(figsize=(8, 5))
plt.plot(turnovers, sharpes, "-o", label="Kalman filter (dynamic β)")
plt.scatter(turnover_static, sharpe_static, c="red", zorder=5, label="Static EG")

plt.xlabel("Average Turnover |Δβ|")
plt.ylabel("Sharpe Ratio")
plt.title("Turnover–Sharpe Frontier")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
