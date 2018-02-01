import pandas as pd
import numpy as np
import simulation as simul
import hayashi_yoshida as hy
import uncertainty_zones as uz
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


df_1 = pd.read_csv('trades_1_simple.csv')
df_2 = pd.read_csv('trades_2_simple.csv')


"""
First test to look at etas and HY naive
eta_1 = 0.11
eta_2 = 0.43
"""
# eta_1 = uz.eta_hat_df(df_1)
# eta_2 = uz.eta_hat_df(df_2)
#
# print("Estimation of eta for ESX =", eta_1)
# print("Estimation of eta for DAX =", eta_2, end="\n")
#
# hy_brut = hy.hayashi_yoshida_lin(df_1, df_2)
# var_1 = hy.hayashi_yoshida_lin(df_1, df_1)
# var_2 = hy.hayashi_yoshida_lin(df_2, df_2)
#
# corr_brut = hy_brut / np.sqrt(var_1 * var_2)
#
# print("Hayashi-Yoshida estimator without efficient prices =", hy_brut)
# print("Correl estimator without efficient prices =", corr_brut)

df_1_last_traded = uz.last_traded_price(df_1)
df_2_last_traded = uz.last_traded_price(df_2)

# print(df_1_last_traded.describe())
# print(df_2_last_traded.describe())
#
# print(df_1_last_traded.head())
# print(df_2_last_traded.head())

# print(df_1_last_traded.loc[0:50,:])
# print(df_2_last_traded.loc[0:50,:])

tick_1 = 1.0
tick_2 = 0.5

eta_1 = 0.11
eta_2 = 0.43

df_1_efficient = uz.efficient_price(df_1_last_traded, tick_1, eta_1)
df_2_efficient = uz.efficient_price(df_2_last_traded, tick_2, eta_2)

hy_eff = hy.hayashi_yoshida_lin(df_1_efficient, df_2_efficient)
var_1 = hy.hayashi_yoshida_lin(df_1_efficient, df_1_efficient)
var_2 = hy.hayashi_yoshida_lin(df_2_efficient, df_2_efficient)

corr_eff = hy_eff / np.sqrt(var_1 * var_2)

print("Hayashi-Yoshida estimator with efficient prices =", hy_eff)
print("Correl estimator with efficient prices =", corr_eff)


def test_sampling_hayashi_yoshida(sample, prices_1, prices_2, microstructure="price"):
    cov = np.zeros(sample.shape)
    var_1 = np.zeros(sample.shape)
    var_2 = np.zeros(sample.shape)
    rho = np.zeros(sample.shape)
    for i, s in enumerate(samples):
        prices_1_sampled = prices_1.iloc[::s]
        prices_2_sampled = prices_2.iloc[::4*s]
        prices_1_sampled.reset_index(inplace=True)
        prices_2_sampled.reset_index(inplace=True)
        cov[i] = hy.hayashi_yoshida_lin(prices_1_sampled, prices_2_sampled, microstructure)
        var_1[i] = hy.hayashi_yoshida_lin(prices_1_sampled, prices_1_sampled, microstructure)
        var_2[i] = hy.hayashi_yoshida_lin(prices_2_sampled, prices_2_sampled, microstructure)
        rho[i] = cov[i] / np.sqrt(var_1[i] * var_2[i])
    plt.plot(sample, var_1)
    plt.plot(sample, var_2)
    plt.plot(sample, cov)
    plt.show()
    plt.plot(sample, rho)
    plt.title('rho sampled')
    plt.show()
    return cov, var_1, var_2


samples = np.arange(1, 20, 2) # , 30, 40, 50, 100, 200, 300, 400, 500, 1000])
cov, var_1, var_2 = test_sampling_hayashi_yoshida(samples, df_1_efficient, df_2_efficient, "price")
plt.plot(samples, var_1)
plt.plot(samples, var_2)




