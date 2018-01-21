"""
Yahoo Finance and computation of the Betas

PCA CAC 40
author: Gregory Calvez
"""

import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
import pandas as pd

ROOT = "/home/gregory/Desktop/Cours_EK/Rosenbaum/"
assets = np.array(pd.read_csv(ROOT + 'cac40.csv', header = None)[[0]][0])
assets
#%%
from bs4 import BeautifulSoup
import urllib3
http = urllib3.PoolManager()


def get_price(name):
    data_open = []
    data_close = []
    url = 'https://finance.yahoo.com/quote/' + name + '/history?period1=1502834400&period2=1510786800&interval=1d&filter=history&frequency=1d'
    #url = "https://finance.yahoo.com/quote/" + name + "/history/"
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data, 'lxml')
    rows = soup.findAll('table')[1].tbody.findAll('tr')
    for each_row in rows:
            divs = each_row.findAll('td')
            if divs[1].span.text  != 'Dividend': #Ignore this row in the table
                #I'm only interested in 'Open' price; For other values, play with divs[1 - 5]
                data_open.append({'Date': divs[0].span.text, 'Open': float(divs[1].span.text.replace(',',''))})
                
                data_close.append({'Date': divs[0].span.text, 'Close': float(divs[4].span.text.replace(',',''))})
    num_periods = len(data_close)
    np_periods = np.zeros(num_periods)
    for i, period in enumerate(data_close):
        np_periods[i] = period['Close']
    return np_periods

#%%
time = len(get_price(assets[0]))
r = np.zeros([assets.size,time])
for i in range(assets.size):
    print(assets[i])
    r[i, :] = get_price(assets[i])
    
#%%
plt.plot(np.arange(time), r[2])
plt.title('Daily price BNP')
plt.show()
#%%
# Une journée défaillante. (à appliquer deux fois)
erreur = np.argmin(r)
r = np.delete(r, erreur, axis = 1)
time -= 1
#%%
num_assets = r.shape[0]
time -= 1
y = np.zeros([num_assets, time])
for i in range(time):
    y[:, i] = r[:, i] / r[:, i+1] - 1
plt.plot(y[2])
plt.title('Daily yield BNP')
plt.show()
#%%
for i in range(num_assets):
    y[i, :] = ( y[i, :] - y[i, :].mean() ) / (y[i, :].std())
#%%
from sklearn.decomposition import *
pca = PCA(n_components = 2)
pca.fit(y.transpose())

print(pca.components_.shape)
#%%
axe = pca.components_[0]
axe /= np.sqrt((axe*axe).sum())
v1 = np.zeros([time])
for i in range(time):
    v1[i] = (axe * y[:, i]).sum() 
#%%   
axe = pca.components_[1]
axe /= np.sqrt((axe*axe).sum())
v2 = np.zeros([time])
for i in range(time):
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

for i, txt in enumerate(assets):
    ax.annotate(txt, (pca.components_[0][i], pca.components_[1][i]), color = 'r')
plt.title('Components of the axis')