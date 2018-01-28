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

"""
returns True if [a,b[ intersects [c,d[
Intervals are considered open to the right to match well with the synchronous case
"""
def intersects(interval1, interval2):
    a, b = interval1
    c, d = interval2
    return a <= d and b > c


def hayashi_yoshida(df_X, df_Y):
    estimator = 0
    df_X['delta_X'] = np.log(df_X['price']) - np.log(df_X['price'].shift(1))
    df_X.fillna(0, inplace=True)

    df_Y['delta_Y'] = np.log(df_Y['price']) - np.log(df_Y['price'].shift(1))
    df_Y.fillna(0, inplace=True)

    i_Y = 1
    n_X = df_X.shape[0]
    n_Y = df_Y.shape[0]
    
    for i_X in np.arange(1, n_X):
        print(i_X / n_X)
        I_X = [df_X['time'][i_X - 1], df_X['time'][i_X]]

        for i_Y in np.arange(1, n_Y):
            I_Y = [df_Y['time'][i_Y - 1], df_Y['time'][i_Y]]
        
            if intersects(I_X, I_Y):
                estimator += df_X['delta_X'][i_X] * df_Y['delta_Y'][i_Y]

    return estimator

def hayashi_yoshida_lin(df_x, df_y): 
    estimator = 0
    df_x['delta_x'] = np.log(df_x['price']) - np.log(df_x['price'].shift(1))
    df_x.fillna(0, inplace=True)
    df_y['delta_y'] = np.log(df_y['price']) - np.log(df_y['price'].shift(1))
    df_y.fillna(0, inplace=True) 
    n_x = df_x.shape[0]
    n_y = df_y.shape[0]
    i_y = 1
    def intersect_index(i_x, i_y):
        if ((i_x < n_x) and (i_y < n_y)): 
            interval_x = [df_x['time'][i_x - 1], df_x['time'][i_x]]
            interval_y = [df_y['time'][i_y - 1], df_y['time'][i_y]]
            return intersects(interval_x, interval_y)
        else: 
            print("Problem index intersect")
    for i_x in np.arange(1, n_x):
        loop = (i_y < n_y)
        while(loop):
            estimator += df_x['delta_x'][i_x] * df_y['delta_y'][i_y]
            i_y += 1
            if(i_y < n_y):
                loop = intersect_index(i_x, i_y)
            else: 
                loop = False
        i_y -= 1
    return estimator