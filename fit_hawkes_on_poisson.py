# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 11:49:40 2018

@author: gregory
"""

"""
Let's test the Hawkes estimation on a Poisson process.
It looks like, no matter what we try to fit w/ a Hawkes, it works... 
"""


import numpy as np
import matplotlib.pyplot as plt
from simulation import *

#%%

intensity = 10000
poisson_process = simulation_poisson(intensity)
#%%

t = np.array([poisson_process])
n = np.array([poisson_process.size])
d = 1
P = 40
t_max = 1*1e-3
tau = t_max/P
h = 1e-3

"""
TODO: 
    We should encapuslate the Hawkes estimation in a class
"""
#%%
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
#%%
"""
Plotting the results
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



"""
Conclusions : 
    In the poisson process estimation as a Hawkes, we find that 
    the exogenous intensity lambda (lam variable / mu in Rosenbaum's slides)
    is far greater than the Hawkes kernel. 
    In comparison, the lamba (lam variable - exogenous intensity) 
    on the DAX Eurostoxx is far smaller than the Hawkes kernel. 
    We hence exhibited the retroactive nature of the market. 
    We could try to determine a lead-lag effet with the kernels alone by 
    fitting exponential or power law on the kernels. And define a "time of 
    response". 
"""