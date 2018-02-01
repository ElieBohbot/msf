import numpy as np
import pandas as pd

def eta_hat_df(df):
    N_total, N_a, N_c = 0, 0, 0
    N = df.shape[0]
    """Let's say we start with an upward move"""
    up = True

    for i in np.arange(1,N):
        if df['price'][i] > df['price'][i-1]:
            if up:
                N_c += 1
            else:
                N_a += 1
                up = True
        if df['price'][i] < df['price'][i-1]:
            if up:
                N_a += 1
                up = False
            else:
                N_c += 1

    return 0.5 * (N_c / N_a)

"""
Another version with numpy arrays as input
More convenient 
"""
def eta_hat(time, price):
    N_total, N_a, N_c = 0, 0, 0
    N = time.shape[0]
    """Let's say we start with an upward move"""
    up = True
    for i in np.arange(1, N):
        if price[i] > price[i-1]:
            if up:
                N_c += 1
            else:
                N_a += 1
                up = True
        if price[i] < price[i-1]:
            if up:
                N_a += 1
                up = False
            else:
                N_c += 1

    return 0.5 * (N_c / N_a)

"""
Retrieving the last traded price of the asset 
"""
def last_traded_price(df):
    times = []
    prices = []
    N = df['time'].shape[0]
    for i in np.arange(1, N):
        if df['price'][i] != df['price'][i-1]:
            times.append(df['time'][i])
            prices.append(df['price'][i])
    times = np.array(times)
    prices = np.array(prices)
    h = np.vstack((times, prices)).transpose()
    df = pd.DataFrame(h, columns=['time', 'price'])
    return df


"""
A method to estimate the efficient price from observed prices, the tick value, and eta_hat
"""
def efficient_prices(last_traded_price, tick, eta_hat):
    def sign(x):
        if x >= 0:
            return 1
        else:
            return -1
    N = last_traded_price.shape[0]
    efficient = np.zeros(N)
    efficient[0] = last_traded_price[0]
    for i in np.arange(1, N):
        efficient[i] = last_traded_price[i] - tick * (0.5 - eta_hat) * sign(last_traded_price[i] - last_traded_price[i - 1])
    return efficient


def efficient_prices_exit(observed_prices, times, tick, eta_hat):
    def sign(x):
        if x >= 0:
            return 1
        else:
            return -1
    exit_times = []
    efficient = []
    N = observed_prices.shape[0]
    for i in np.arange(1, N):
        if observed_prices[i] != observed_prices[i-1]:
            exit_times.append(times[i])
            efficient.append(observed_prices[i] - tick * (0.5 - eta_hat) * sign(observed_prices[i] - observed_prices[i-1]))

    exit_times = np.array(exit_times)
    efficient = np.array(efficient)
    df = pd.DataFrame(np.vstack((exit_times, efficient)).transpose(), columns=["time", "efficient_price"])
    return df



"""
Getting efficient price from last traded price
- Problem : I don't understand how to get the exit time (called tau_i in Rosenbaum's slides)
- the exit times are necessary for the HY estimator
- making the assumption that tau_i = t_i (even if not really realistic)
"""
def efficient_price(df, tick, eta_hat):
    def sign(x):
        if x >= 0:
            return 1
        else:
            return -1
    N = df.shape[0]
    efficient = []
    efficient.append(df['price'][0])
    for i in np.arange(1, N):
        efficient.append(df['price'][i] - tick * (0.5 - eta_hat) * sign(df['price'][i] - df['price'][i-1]))
    times = np.array(df['time'])
    efficient = np.array(efficient)
    h = np.vstack((times, efficient)).transpose()
    df = pd.DataFrame(h, columns=['time', 'price'])
    return df
