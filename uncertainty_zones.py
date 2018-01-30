import numpy as np

def eta_hat_df(df):
    N_total, N_a, N_c = 0, 0, 0
    N = df.shape[0]
    """Let's say we start with an upward move"""
    up = True

    for i in np.arange(1,N):
        if df['price_tick'][i] > df['price_tick'][i-1]:
            if up:
                N_c += 1
            else:
                N_a += 1
                up = True
        if df['price_tick'][i] < df['price_tick'][i-1]:
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
A method to estimate the efficient price from observed prices, the tick value, and eta_hat
"""
def efficient_prices(observed_prices, tick, eta_hat):
    def sign(x):
        if x >= 0:
            return 1
        else:
            return -1
    N = observed_prices.shape[0]
    efficient = np.zeros(N)
    efficient[0] = observed_prices[0]
    for i in np.arange(1, N):
        efficient[i] = observed_prices[i] - tick * (0.5 - eta_hat) * sign(observed_prices[i] - observed_prices[i-1])
    return efficient