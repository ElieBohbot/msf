# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:03:48 2018

@author: gregory
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#test = 500
"""
Random times
First let try to creat a Poisson process
"""
def simulation_poisson(intensity):
    """
    Simulation of a Poisson process. 
    Args: 
            intensity(float): intensity of the process
    """
    n = intensity * 3
    x = np.random.exponential(1./intensity, round(n)).cumsum()
    times = pd.DataFrame(x, columns=["Time"])
    times = times[times["Time"] < 1]
    return np.array(times["Time"])
#times = simulation_poisson(test)

def brownian(x):
    """
    Simulation of a brownian motion given the times of the process
    """
    n = x.size
    covariance = np.zeros([n,n])
    for i in range(n):
        # It can be done thanks to numpy in one loop but too complicated 
        for j in range(n):
            covariance[i,j] = min(x[i], x[j])
    a = np.linalg.cholesky(covariance)
    mb = a.dot(np.random.randn(n))
    return mb
#mb = brownian(times)

def correlated_brownians(intensity_1, intensity_2, rho):
    """
    Created two correlated brownian motion sampled at random times
    Args: 
        rho(float): b/w -1 and 1, correlation
        intensity(float): >0, intensity of each Poisson process
    """
    times_1 = simulation_poisson(intensity_1)
    times_2 = simulation_poisson(intensity_2)
    times = np.concatenate((times_1, times_2))
    times.sort()
    x_1 = brownian(times)
    x_2 = brownian(times)
    w_1 = x_1
    w_2 = rho * x_1 + np.sqrt(1 - rho**2) * x_2
    w_1_sampled = list()
    w_2_sampled = list()
    for i,t in enumerate(times):
        if (t in times_1):
            w_1_sampled.append(w_1[i])
        if (t in times_2):
            w_2_sampled.append(w_2[i])
    return times_1, np.array(w_1_sampled), times_2, np.array(w_2_sampled)
#times_1, w_1, times_2, w_2 = correlated_brownians(100, 300, 0.4)
#plt.plot(times_1, w_1, 'b')
#plt.plot(times_2, w_2, 'r')
#%%
def round_tick(price, tick):
    """
    Round prices to be on ticks
    """
    return np.round(price / tick) * tick


def black_scholes(intensity_1, intensity_2, rho, vol_1, vol_2, s_1, s_2, tick_1, tick_2):
    """
    Create two black scholes simulation with chosen parameters and sampled
    thanks to Poisson process. The values are rounded to the closest tick. 
    The tick should be chosen carefully. 
    Args: 
        intensity(float): >0
        rho(float): -1<rho<1
        vol(float): >0 volatility
        s(float): >0 initial values
        tick(float)
    """
    times_1, w_1, times_2, w_2 = correlated_brownians(intensity_1, intensity_2, rho)
    price_1 = s_1 * np.exp(-vol_1**2 / 2 * times_1 + vol_1 * w_1)
    price_2 = s_2 * np.exp(-vol_2**2 / 2 * times_2 + vol_2 * w_2)
    price_1_rounded = round_tick(price_1, tick_1)
    price_2_rounded = round_tick(price_2, tick_2)
    return times_1, price_1, price_1_rounded, times_2, price_2, price_2_rounded

def black_scholes_df(intensity_1, intensity_2, rho, vol_1, vol_2, s_1, s_2, tick_1, tick_2):
    times_1, price_1, price_1_rounded, times_2, price_2, price_2_rounded = black_scholes(intensity_1, intensity_2, rho,
                                                                                         vol_1, vol_2, s_1, s_2, tick_1,
                                                                                         tick_2)
    df_1 = pd.DataFrame(np.vstack((times_1, price_1, price_1_rounded)).transpose(), columns=['time', 'price', 'price_tick'])
    # df_1.set_index('time', inplace=True)
    df_2 = pd.DataFrame(np.vstack((times_2, price_2, price_2_rounded)).transpose(), columns=['time', 'price', 'price_tick'])
    # df_2.set_index('time', inplace=True)
    return df_1, df_2


# times_1, w_1, times_2, w_2 = black_scholes(100, 200, 0.4, 0.02, 0.03, 1000, 1000, 2, 3)
# plt.plot(times_1, w_1, 'b')
# plt.plot(times_2, w_2, 'r')
#%%
"""
Let's try to plot the prices like in Rosenbaum's course

"""
def plot(times, price, n):
    tt = np.linspace(0, 1, n)
    x = np.zeros(n)
    count = 0
    for i, t in enumerate(tt):
        print(i, t)
        while (count < times.size - 2) and (times[count]<t) : 
            count+=1
        x[i] = price[count]
    plt.figure()
    plt.plot(tt, x)
    plt.show()

# plot(times_1, w_1, 1000)
# plot(times_2, w_2, 1000)



"""
Adding a few methods to simulate synchronous data
"""

def sync_correlated_brownians(intensity, rho):
    """
    Created two correlated brownian motion sampled at random times
    Args:
        rho(float): b/w -1 and 1, correlation
        intensity(float): >0, intensity of each Poisson process
    """
    times= simulation_poisson(intensity)
    x_1 = brownian(times)
    x_2 = brownian(times)
    w_1 = x_1
    w_2 = rho * x_1 + np.sqrt(1 - rho**2) * x_2
    return times, w_1, w_2


def sync_black_scholes(intensity, rho, vol_1, vol_2, s_1, s_2, tick_1, tick_2):
    """
    Create two black scholes simulation with chosen parameters and sampled
    thanks to Poisson process. The values are rounded to the closest tick.
    The tick should be chosen carefully.
    Args:
        intensity(float): >0
        rho(float): -1<rho<1
        vol(float): >0 volatility
        s(float): >0 initial values
        tick(float)
    """
    times, w_1, w_2 = sync_correlated_brownians(intensity, rho)
    price_1 = s_1 * np.exp(-vol_1**2 / 2 * times + vol_1 * w_1)
    price_2 = s_2 * np.exp(-vol_2**2 / 2 * times + vol_2 * w_2)
    price_1_rounded = round_tick(price_1, tick_1)
    price_2_rounded = round_tick(price_2, tick_2)
    return times, price_1, price_1_rounded, price_2, price_2_rounded


def sync_black_scholes_df(intensity, rho, vol_1, vol_2, s_1, s_2, tick_1, tick_2):
    times, price_1, price_1_rounded, price_2, price_2_rounded = sync_black_scholes(intensity, rho, vol_1, vol_2, s_1, s_2, tick_1, tick_2)
    df_1 = pd.DataFrame(np.vstack((times, price_1, price_1_rounded)).transpose(), columns=['time', 'price', 'price_tick'])
    # df_1.set_index('time', inplace=True)
    df_2 = pd.DataFrame(np.vstack((times, price_2, price_2_rounded)).transpose(), columns=['time', 'price', 'price_tick'])
    # df_2.set_index('time', inplace=True)
    return df_1, df_2


