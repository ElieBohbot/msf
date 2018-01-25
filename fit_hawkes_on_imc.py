# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:09:13 2018

@author: gregory
"""
"""
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from pandas import HDFStore,DataFrame

root = "/home/gregory/Desktop/pycontest-master-1064c0b51d031c8f98263eb29db008b113cf91a7/"
file_name = "Training_20151109.h5"

hdf = pd.HDFStore(root + file_name)

expiry_indec = pd.read_hdf(root + file_name, "expiryIndex")
fund_index = pd.read_hdf(root + file_name, "fundIndex")
product_data = pd.read_hdf(root + file_name, "productData")
feed_states = pd.read_hdf(hdf, "feedStates")


feed_states.head()
feed_states.columns
feed_states['type'][0]

print(product_data)
book_id_1 = product_data['bookIdNum'][0]
book_id_2 = product_data['bookIdNum'][1]

product_1 = feed_states[feed_states['bookIdNum'] == book_id_1]
product_2 = feed_states[feed_states['bookIdNum'] == book_id_2]

bid_price_1 = product_1['Bid_1']
ask_price_1 = product_1['Ask_1']
begin = 1000000
end = begin + 6000
plt.plot(bid_price_1[begin:end])
plt.plot(ask_price_1[begin:end], 'r')

product_1["type"].describe()

product_1_type_1 = product_1[product_1["type"] == 1]
product_1_type_2 = product_1[product_1["type"] == 2]
product_1_type_3 = product_1[product_1["type"] == 3]

x1 = product_1_type_1[:100]
x2 = product_1_type_2[: 100]
x3 = product_1_type_3[: 100]

plt.plot(product_1_type_2["price"])
plt.plot(product_2[product_2["type"]==2]["price"])
"""
#%%

"""
Let's have some clean data
We keep only the trades : volume, price, agressor side (even if I do not 
know what this means), timestamps and times
"""

import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
from matplotlib import style

root = "/home/gregory/Desktop/pycontest-master-1064c0b51d031c8f98263eb29db008b113cf91a7/"
file_name = "Training_20151109.h5"

hdf = pd.HDFStore(root + file_name)

expiry_indec = pd.read_hdf(root + file_name, "expiryIndex")
fund_index = pd.read_hdf(root + file_name, "fundIndex")
product_data = pd.read_hdf(root + file_name, "productData")
feed_states = pd.read_hdf(hdf, "feedStates")


trades_1 = feed_states[feed_states["bookIdNum"] == product_data['bookIdNum'][0]]
trades_1 = trades_1[trades_1["type"] == 2]

trades_2 = feed_states[feed_states["bookIdNum"] == product_data['bookIdNum'][1]]
trades_2 = trades_2[trades_2["type"] == 2]


fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(trades_1["Time"], trades_1["price"], 'b')
ax1.set_ylabel('1', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(trades_2["Time"], trades_2["price"], 'r')
ax2.set_ylabel('2', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
    
    
#%% Hawkes    
time_1 = trades_1["Time"]
time_2 = trades_2["Time"]
t_max = max(time_1.max(), time_2.max())
t_min = min(time_1.min(), time_2.min())

def rescale_time(time, t_max, t_min):
    return (time-t_min) / (t_max-t_min)
time_1_rescale = rescale_time(time_1, t_max, t_min)
time_2_rescale = rescale_time(time_2, t_max, t_min)
mini = 0
maxi = 1
time_1_rescale = time_1_rescale[time_1_rescale<maxi]
time_2_rescale = time_2_rescale[time_2_rescale<maxi]

time_1_rescale = time_1_rescale[time_1_rescale > mini]
time_2_rescale = time_2_rescale[time_2_rescale > mini]

#%%
"""
Script de l'estimation non paramétrique du noyau
"""
t1 = np.array(time_1_rescale)
t2 = np.array(time_2_rescale)
d = 2
n = np.array([len(t1), len(t2)])
t = np.array([t1, t2])


"""
Paramètres de l'estimation. Ces paramètres sont proposés dans la littérature
"""

P = 40
t_max = 3*1e-7
tau = t_max/P
h = 1e-6


def compute_lam():
    l = np.zeros(d)
    for i in range(d):
        l[i] = n[i]/t[i][-1]
    return l
lam = compute_lam()
def K(x):
    return (0<=x) * (x<=1)
def gauss(x, moy, sigma):
    return 1/np.sqrt(2*3.14*sigma**2) * np.exp(- (x-moy)**2 / (2*sigma))
def add_dirac(f, t, tau, h):
    for s in range(f.size):
        x = t-s*tau
        if ((0<x) and (x<h)):
            f[s] += 1/h
def dirac(t_max, h, P):
    tau = t_max/P
    d = np.zeros(P)
    d[0]=1/h
    return d
    
#%%
def compute_g(h, t_max, P):
    g = np.zeros([d, d, P])
    tau = t_max / P
    for i in range(d):
        for j in range(d):
            compt = 0
            index_i = 0
            for tj in t[j]:
                if (compt % 100 == 0):
                    print(compt)
                compt+=1
                index_i_sec = index_i
                ti = t[i][index_i_sec]
                while((index_i_sec < n[i]-1) and (ti < tj-h)):
                    index_i_sec += 1
                    ti = t[i][index_i_sec]
                index_i = index_i_sec
                index_i_prime = index_i_sec
                while((index_i_prime < n[i]-1) and (t[i][index_i_prime] < tj + t_max+h)):
                    ti = t[i][index_i_prime]
                    if (ti!=tj):
                        add_dirac(g[i, j, :], ti-tj, tau, h)
                    index_i_prime += 1
            g[i, j, :] = g[i, j, :] / n[j]
            if (i==j):
                g[i, j, :] = g[i, j, :]  #- dirac(t_max, h, P)
        g[i, :, :] -= lam[i] * np.outer(np.ones(d), np.ones(P))
    return g
print("Computing g")
g = compute_g(h, t_max, P)
def reshape(y):
    x = np.zeros([d, P])
    for i in range(d):
        for p in range(P):
            x[i, p] = y[i*P + p]
    return x
def melt(x):
    y = np.zeros([d*P])
    for i in range(d):
        for p in range(P):
            y[i*P + p] = x[i, p]
    return y
def K_prime(i, j, k, pq):
    if (pq>0):
        return g[k,j,pq]
    elif (pq<0):
        return g[j,k,-pq] * lam[k]/lam[j]
    else:
        return 0
def compute_phi():
    phi = np.zeros([d, d, P])
    for i in range(d):
        M = np.zeros([d*P, d*P])
        for j in range(d):
            for p in range(P):
                for k in range(d):
                    for q in range(P):
                        if ((p==q)and(j==k)):
                            M[j*P + p, k*P + q] = 1 + g[j, k, 0]*tau
                        else:
                            M[j*P + p, k*P + q] = K_prime(i, j, k, p-q)*tau
        y = melt(g[i, :, :])
        x = np.linalg.inv(M).dot(y)
        phi[i, :, :] = reshape(x)
    return phi, M
print("Computiing phi")
phi, M = compute_phi()


""" 
Plot des résultats
"""
x = np.arange(P)*tau
for i in range(d):
    for j in range(d):
        plt.plot(x, phi[i, j, :], label = 'phi'+str(i+1) + str(j+1))
        plt.legend()
plt.xlabel('Time)')
plt.ylabel('Kernel')
plt.title('Estimated Kernel')
plt.show()