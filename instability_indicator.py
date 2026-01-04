import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


np.random.seed(0)
T = 1000
price_x = np.exp(np.cumsum(np.random.normal(0, 0.01, T)))
price_y = np.exp(np.cumsum(np.random.normal(0, 0.01, T)))
x = pd.Series(np.log(price_x))
y = pd.Series(np.log(price_y))


X = sm.add_constant(x.values)
beta_static = sm.OLS(y.values, X).fit().params[1]
beta_static_series = np.full(T, beta_static)


def kalman_beta_gain(x, y, Q, R=1e-4):
    T = len(x)
    beta = np.zeros(T)
    P = np.zeros(T)
    K_arr = np.zeros(T)

    beta[0] = 0.0
    P[0] = 1.0

    for t in range(1, T):
        beta_pred = beta[t-1]
        P_pred = P[t-1] + Q

        H = x.iloc[t]
        S = H**2 * P_pred + R
        K = P_pred * H / S

        beta[t] = beta_pred + K * (y.iloc[t] - H * beta_pred)
        P[t] = (1 - K * H) * P_pred
        K_arr[t] = K

    return beta, K_arr, P

Q_small = 1e-2  
beta_kf, K_arr, P_arr = kalman_beta_gain(x, y, Q_small)

spread = y - beta_kf * x
spread_stat=y-beta_static*x

fig, ax1 = plt.subplots(figsize=(12,5))

ax1.plot(beta_kf, label='Dynamic β', color='blue')

ax1.set_xlabel('Time')
ax1.set_ylabel('β / Spread')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(K_arr, label='Kalman gain |K_t|', color='red', alpha=0.5)
ax2.set_ylabel('Kalman Gain')
ax2.legend(loc='upper right')

plt.title('Dynamic β, and Kalman Gain over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
