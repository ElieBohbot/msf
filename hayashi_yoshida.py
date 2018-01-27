import numpy as np
import pandas as pd


def basic_estimator(df_X, df_Y):
    df_X['delta_X'] =  np.log(df_X['price']) - np.log(df_X['price'].shift(1))
    df_X.fillna(0, inplace=True)

    df_Y['delta_Y'] =  np.log(df_Y['price']) - np.log(df_Y['price'].shift(1))
    df_Y.fillna(0, inplace=True)

    c = np.sum(df_X['delta_X'] * df_Y['delta_Y'])
    den = np.sqrt(np.sum((df_X.delta_X)**2) * np.sum((df_Y.delta_Y)**2))

    return c / den

def intersects(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return a <= d and b >= c


def hayashi_yoshida(df_X, df_Y):
    estimator = 0
    df_X['delta_X'] = np.log(df_X['price']) - np.log(df_X['price'].shift(1))
    df_X.fillna(0, inplace=True)

    df_Y['delta_Y'] = np.log(df_Y['price']) - np.log(df_Y['price'].shift(1))
    df_Y.fillna(0, inplace=True)

    i_Y = 1
    n = df_X.shape[0]

    for i_X in np.arange(1, n):

        I_X = [df_X['time'][i_X - 1], df_X['time'][i_X]]
        I_Y = [df_Y['time'][i_Y - 1], df_Y['time'][i_Y]]
        
        while intersects(I_X, I_Y) and i_Y < n - 1:
            estimator += df_X['delta_X'][i_X] * df_Y['delta_Y'][i_Y]
            i_Y += 1
            I_Y = [df_Y['time'][i_Y - 1], df_Y['time'][i_Y]]

    return estimator


#returns a random subdivision of [0,1] of size _size_
def random_sample(size):
    time = np.random.random_sample(size)
    return time.cumsum() / np.max(time.cumsum())


# size = 1000
#
# time_X = random_sample(size)
# time_Y = random_sample(size)
#
# X = 10 + np.random.randn(size)
# Y = 10 + np.random.randn(size)
# df_X = pd.DataFrame({'time':time_X, 'X':X})
# df_Y = pd.DataFrame({'time':time_Y, 'Y':Y})
#
# print(hayashi_yoshida(df_X, df_Y))
# basic_estimator(df_X, df_Y)
