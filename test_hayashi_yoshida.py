# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:11:51 2018

@author: gregory
"""

from hayashi_yoshida import *
from uncertainty_zones import *  

prices_1 = trades_1[["Time","price"]]
prices_2 = trades_2[["Time", "price"]]

prices_1.columns = ["time", "price"]
prices_2.columns = ["time", "price"]

eta_hat_1 = eta_hat(np.array(prices_1["time"]), np.array(prices_1["price"]))
eta_hat_2 = eta_hat(np.array(prices_2["time"]), np.array(prices_2["price"]))

prices_1["efficient"] = efficient_prices(np.array(prices_1["price"]), 1, eta_hat_1)
prices_2["efficient"] = efficient_prices(np.array(prices_2["price"]), 0.5, eta_hat_2)


#%%
def test_sampling_hayashi_yoshida(sample, prices_1, prices_2, microstructure = "price"):
    cov = np.zeros(sample.shape)
    var_1 = np.zeros(sample.shape)
    var_2 = np.zeros(sample.shape)
    rho = np.zeros(sample.shape)
    for i, s in enumerate(samples):
        prices_1_sampled = prices_1.iloc[::s]
        prices_2_sampled = prices_2.iloc[::s]
        prices_1_sampled.reset_index(inplace = True)
        prices_2_sampled.reset_index(inplace = True)
        cov[i] = hayashi_yoshida_lin(prices_1_sampled, prices_2_sampled, microstructure)
        var_1[i] = hayashi_yoshida_lin(prices_1_sampled, prices_1_sampled, microstructure)
        var_2[i] = hayashi_yoshida_lin(prices_2_sampled, prices_2_sampled, microstructure)
        rho[i] = cov[i] / np.sqrt(var_1[i] * var_2[i])
    plt.plot(sample, var_1)
    plt.plot(sample, var_2)
    plt.plot(sample, cov)
    plt.show()
    plt.plot(sample, rho)
    plt.show()
    return cov, var_1, var_2
    
samples = np.arange(1, 50)*2#, 30, 40, 50, 100, 200, 300, 400, 500, 1000])
cov, var_1, var_2 = test_sampling_hayashi_yoshida(samples, prices_1, prices_2, "price")
plt.plot(samples, var_1)
plt.plot(samples, var_2)

#%%
"""
Autocorrélation
"""
scale = 1e9
def autocorrel(prices_or, delta, microstructure, sample = 1):
    prices = prices_or.copy()
    prices_shifted = prices.copy()
    prices_shifted["time"] += delta
    prices = prices.iloc[::sample]
    prices_shifted = prices_shifted.iloc[::sample]
    prices.reset_index(inplace = True)
    prices_shifted.reset_index(inplace = True)
    return hayashi_yoshida_lin(prices, prices_shifted, microstructure)
ddelta = np.linspace(-600, 600, 31) * scale
autocor = np.zeros(ddelta.shape)
for i, delta in enumerate(ddelta):
    autocor[i] = autocorrel(prices_2, delta, "efficient")
plt.plot(ddelta, autocor, '+')
#%%
scale = 1e9
def lead_lag(prices_or, prices_shifted_or, delta, microstructure, sample = 60):
    prices = prices_or.copy()
    prices_shifted = prices_shifted_or.copy()
    prices_shifted["time"] += delta
    prices = prices.iloc[::sample]
    prices_shifted = prices_shifted.iloc[::sample]
    prices.reset_index(inplace = True)
    prices_shifted.reset_index(inplace = True)
    return hayashi_yoshida_lin(prices, prices_shifted, microstructure)
ddelta = np.linspace(-3, 3, 51) * scale
covar = np.zeros(ddelta.shape)
for i, delta in enumerate(ddelta):
    covar[i] = lead_lag(prices_1, prices_2, delta, "price")
plt.plot(ddelta/1e9, covar, '+')
plt.xlabel('Delta (sec)')
plt.ylabel('Covariance')
print(ddelta[covar.argmax()]/1e9)
#%%
"""
We should try to identify some basics HF stylized fact. 
First of all, the U shape vol. Or equivalently, the change of duration 
during the day.
"""
prices_1["duration"] = prices_1["time"] - prices_1["time"].shift(1)

for 
plt.plot(prices_1_sampled["time"][1:], prices_1_sampled["duration"][1:])