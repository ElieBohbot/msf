# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 22:57:40 2017

@author: gregory
"""

"""
Yahoo Finance and computation of the Betas

Test for the CAPM
"""

import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
import pandas as pd
from sklearn.decomposition import *

ROOT = "/home/gregory/Desktop/Cours_EK/Rosenbaum/Project/msf/"
file_name = "sp500_joined_closes.csv"
data_pd = pd.read_csv(ROOT + file_name)
date = data_pd['Date']
asset = list(data_pd.columns.values)
col_drop = data_pd.select_dtypes(include = [object]).columns
for col in col_drop:
    del data_pd[col]

data = data_pd.as_matrix()
num_dates_data = data.shape[0]
num_assets = data.shape[1]
"""
Here data is the matrix of correct values for s&p500
504 date (daily)
495 assets
"""
#%%
data = data.transpose()
num_dates = num_dates_data - 1
y = np.zeros([num_assets, num_dates])
for i in range(num_dates):
    y[:, i] = data[:, i+1] / data[:, i] - 1
#%%
k = 300
plt.plot(y[k, :])
plt.title(asset[k])
#%%
time = len(get_yield(assets[0]))
r = np.zeros([assets.size,time])
for i in range(assets.size):
    print(assets[i])
    r[i, :] = get_yield(assets[i])
r_marche = get_yield('^FCHI')
r_riskless = (1 - 0.6021)**(1/52) - 1
#%%
z = r - r_riskless
z_marche = r_marche - r_riskless
#%%
plt.hist(z_marche)
plt.title('Rendements Nets hebdo CAC')
plt.show()

plt.hist(z[3])
plt.title('Rendements Nets hebdo BNP')
plt.show()

#%%
mu_hat = z.mean(axis = 1)
mu_hat_star = z_marche.mean()
sigma_2_hat = ((z_marche - mu_hat_star)**2).mean()

beta_hat = ((z.transpose()-mu_hat).transpose()*(z_marche - mu_hat_star)).mean(axis = 1) / sigma_2_hat
alpha_hat = mu_hat - beta_hat*mu_hat_star

#%%
for i in range(assets.size):
    print(assets[i], "   \t beta: ", beta_hat[i], "    \t alpha:", alpha_hat[i])
#%%
import scipy.stats as ss
N = alpha_hat.size
Sigma_hat = np.zeros([N, N])
for t in range(time):
    Sigma_hat += np.outer((z[:, t] - alpha_hat - beta_hat * z_marche[t]),(z[:, t] - alpha_hat - beta_hat * z_marche[t]).transpose())
Sigma_hat /= time
print(Sigma_hat)
Sigma_hat_inv = np.linalg.pinv(Sigma_hat)
tt = 1 / (1 + mu_hat_star**2 / sigma_2_hat) * time * alpha_hat.transpose().dot(Sigma_hat_inv.dot(alpha_hat))
pvalue = ss.t.sf(np.abs(tt), N) * 2
print("Stat: ", tt, "\t pvalue; ", pvalue)


















#%%


import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
n = 10000
data = np.random.uniform(0, 1, [2, n])
x = 1.0 * ((data[0, :]**2 + data[1, :]**2) >= 1)
x *= (data[1, :] <= (2-np.sqrt(3)) * data[0, :] + np.sqrt(3) - 1)
x *= (data[1, :] >= 1/(2-np.sqrt(3)) * data[0, :] - (np.sqrt(3) - 1) / (2-np.sqrt(3)) )
print(x.sum()/n)

p = data*x
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()

plt.scatter(p[0, :], p[1, :])
