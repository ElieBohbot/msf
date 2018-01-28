import numpy as np

def eta_hat(df):
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