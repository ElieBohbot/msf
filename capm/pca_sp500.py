"""
Yahoo Finance and computation of the Betas

PCA CAC 40
author: Gregory Calvez
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
for i in range(num_assets):
    y[i, :] = ( y[i, :] - y[i, :].mean() ) / (y[i, :].std())
y = np.nan_to_num(y)
#%%
pca = PCA(n_components = 2)
pca.fit(y.transpose())

print(pca.components_.shape)
#%%
axe = pca.components_[0]
axe /= np.sqrt((axe*axe).sum())
v1 = np.zeros([num_dates])
for i in range(num_dates):
    v1[i] = (axe * y[:, i]).sum() 
#%%   
axe = pca.components_[1]
axe /= np.sqrt((axe*axe).sum())
v2 = np.zeros([num_dates])
for i in range(num_dates):
    v2[i] = (axe * y[:, i]).sum()

#%%
plt.plot(v1, v2, '+')
plt.axhline(0, color = 'black')
plt.axvline(0, color = 'black')
plt.title('Days projected')
plt.show()
#%%
fig, ax = plt.subplots()
ax.scatter(pca.components_[0], pca.components_[1])

ax.axhline(0, color = 'black')
ax.axvline(0, color = 'black')

for i, txt in enumerate(asset[:50]):
    ax.annotate(txt, (pca.components_[0][i], pca.components_[1][i]), color = 'r')
plt.title('Components of the axis')