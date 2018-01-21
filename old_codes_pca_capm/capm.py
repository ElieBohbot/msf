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
from yahoo_finance import Share
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


def get_yield(name):
    data_open = []
    data_close = []
    url = 'https://finance.yahoo.com/quote/' + name + '/history?period1=1493157600&period2=1508968800&interval=1wk&filter=history&frequency=1wk'
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
    num_weeks = len(data_close)
    np_close = np.zeros(num_weeks)
    for i, week in enumerate(data_close):
        np_close[i] = week['Close']
        
    y = np.exp(-np.diff(np.log(np_close)))
    r = y - 1
    return r

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