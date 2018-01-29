# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:11:51 2018

@author: gregory
"""

from hayashi_yoshida import * 

prices_1 = trades_1[["Time","price"]]
prices_2 = trades_2[["Time", "price"]]

prices_1.columns = ["time", "price"]
prices_2.columns = ["time", "price"]

def test_sampling_hayashi_yoshida(sample):
    cov = np.zeros(sample.shape)
    var_1 = np.zeros(sample.shape)
    var_2 = np.zeros(sample.shape)
    rho = np.zeros(sample.shape)
    for i, s in enumerate(samples):
        prices_1_sampled = prices_1.iloc[::s]
        prices_2_sampled = prices_2.iloc[::s]
        prices_1_sampled.reset_index(inplace = True)
        prices_2_sampled.reset_index(inplace = True)
        cov[i] = hayashi_yoshida_lin(prices_1_sampled, prices_2_sampled)
        var_1[i] = hayashi_yoshida_lin(prices_1_sampled, prices_1_sampled)
        var_2[i] = hayashi_yoshida_lin(prices_2_sampled, prices_2_sampled)
        rho[i] = cov[i] / np.sqrt(var_1[i] * var_2[i])
    plt.plot(sample, var_1)
    plt.plot(sample, var_2)
    plt.plot(sample, cov)
    plt.show()
    plt.plot(sample, rho)
    plt.show()
    return cov, var_1, var_2
    
samples = np.array([1, 5, 10, 20, 40, 100, 1000])
cov, var_1, var_2 = test_sampling_hayashi_yoshida(samples)



#%%

"""
We should try to identify some basics HF stylized fact. 
First of all, the U shape vol. Or equivalently, the change of duration 
during the day.
"""
prices_1["duration"] = prices_1["time"] - prices_1["time"].shift(1)

for 
plt.plot(prices_1_sampled["time"][1:], prices_1_sampled["duration"][1:])