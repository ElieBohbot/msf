import simulation as simul
from matplotlib import style
import matplotlib.pyplot as plt
import hayashi_yoshida as hy
import uncertainty_zones as uz
import numpy as np

"""
Different styles for plotting
"""
# style.use('seaborn-bright')
style.use('bmh')
# style.use('fivethirtyeight')



"""
Test of brownian trajectories
- One single trajectory
- Two correlated trajectories
- BS trajectories
"""

# intensity = 300.0
# time = simul.simulation_poisson(intensity)
# bm = simul.brownian(time)
# plt.plot(time, bm)
# plt.show()

# times_1, w_1, times_2, w_2 = simul.correlated_brownians(300, 300, -0.9)
# plt.plot(times_1, w_1, 'b')
# plt.plot(times_2, w_2, 'r')
# plt.show()


# vol_1 = 0.1
# vol_2 = 0.2
# rho = 0.1
# intensity = 300
# s_0 = 100
# times_1, w_1, w_1_rounded, times_2, w_2, w_2_rounded = simul.black_scholes(intensity, intensity, rho,
#                                                               vol_1, vol_2, s_0, s_0, 2, 3)
# plt.plot(times_1, w_1, 'b')
# plt.plot(times_2, w_2, 'r')
# plt.show()

"""
Test of Hayashi-Yoshida estimator
- Case 1 : Black-Scholes model without asynchronicity 
- Case 2 : Black-Scholes model with asynchronicity 
"""

"""
Case 1
Basic estimator and HY in synchronous case
Both seem to work well

Question : how to compute std's of estimators?
"""
# vol_1 = 0.1
# vol_2 = 0.02
# rho = 0.9
# intensity = 300
# s_0 = 100
#
# df_X, df_Y = simul.sync_black_scholes_df(intensity, rho, vol_1, vol_2, s_0, s_0, 0.1, 0.01)

# print(df_X.head())
# print(df_X.tail())
# print(df_Y.head())
# print(df_Y.tail(), "\n")
#
# df_X.set_index('time', inplace=True)
#
# plt.figure()
# df_X['price'].plot()
#
# plt.step(df_X.index, df_X['price_tick'], 'r', linewidth=0.7)
#
# plt.show()
#
# M = 50
# m = []
# for i in range(M):
#     df_X = simul.sync_black_scholes_df(intensity, rho, vol_1, vol_2, s_0, s_0, 0.1, 0.01)[0]
#     m.append(uz.eta_hat(df_X))
#
# print('mean =', np.mean(m))
# print('std =', np.std(m))
#
# print('Exact value :', rho, "\n")
#
#
# estim = hy.basic_estimator(df_X, df_Y)
# print('Value of rho by basic estimator :', estim)
#
# estim_cov_hy = hy.hayashi_yoshida(df_X, df_Y)
# print('Value of rho by Hayashi-Yoshida :', estim_cov_hy/(vol_1*vol_2))


"""
Case 2:
Asynchronous case
"""

# vol_1 = 0.1
# vol_2 = 0.2
# rho = 0.6
# intensity_1 = 500
# intensity_2 = 450
# s_0 = 100
#
# df_X, df_Y = simul.black_scholes_df(intensity_1, intensity_2, rho, vol_1, vol_2, s_0, s_0, 2, 3)
#
# print('Exact value of covariance:', rho * vol_2 * vol_1, "\n")
#
# estim_cov_hy = hy.hayashi_yoshida(df_X, df_Y)
# print('Value of rho by Hayashi-Yoshida :', estim_cov_hy)
vol_1 = 0.1
vol_2 = 0.2
rho = 0.6
intensity_1 = 200
intensity_2 = 150
s_0 = 100

# df_X, df_Y = simul.black_scholes_df(intensity_1, intensity_2, rho, vol_1, vol_2, s_0, s_0, 2, 3)
# #estim_cov_hy = 0
# estim_cov_hy = hy.hayashi_yoshida(df_X, df_Y)
# estim_cov_hy_lin = hy.hayashi_yoshida_lin(df_X, df_Y)
#
#
# print('Exact value of covariance:', rho * vol_2 * vol_1, "\n")
# print('Value of rho by Hayashi-Yoshida :', estim_cov_hy)
# print('Value of rho by Hayashi-Yoshida lin :', estim_cov_hy_lin)


M = 30
m = []
for i in range(M):
    df_X, df_Y = simul.black_scholes_df(intensity_1, intensity_2, rho, vol_1, vol_2, s_0, s_0, 2, 3)
    m.append(hy.hayashi_yoshida_lin(df_X, df_Y))

print('mean =', np.mean(m))
print('std =', np.std(m))
