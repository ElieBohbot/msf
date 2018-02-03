import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import scipy.stats as sts

style.use('ggplot')

"""
start and end dates of the data
"""
start = datetime(2016, 1, 1)
end = datetime(2017, 12, 31)

df_sp500 = pd.read_csv('sp500_joined_closes.csv')
df_ir = pd.read_csv('ir_10y_treasury.csv')
df_index = pd.read_csv('sp500_index.csv')

"""
Removing corrupted data
"""
time = df_sp500['Date'].shape[0]

df_sp500.dropna(thresh = time - 50, axis=0, inplace=True)
df_sp500.dropna(thresh = time - 50, axis=1, inplace=True)

df_sp500.fillna(method='ffill', inplace=True)

"""
Putting all the data together and creating the matrix qui va bien
"""
"""
First step : dropping useless columns for IR and Index and joining them to SP500 assets
"""
df_ir.rename(columns={'Adj Close': 'IR10Y'}, inplace=True)
df_ir.drop(['Open', 'High', 'Low', 'Close', 'Volume', '100ma'], 1, inplace=True)
df_ir['IR10Y'] = df_ir['IR10Y'] / 100.

df_index.rename(columns={'Adj Close': 'SP500'}, inplace=True)
df_index.drop(['Open', 'High', 'Low', 'Close', 'Volume', '100ma'], 1, inplace=True)

df_data = pd.concat([df_sp500, df_ir, df_index], axis=1, join='inner')
df_data.fillna(method='ffill', inplace=True)

"""
Second step : computing the yields for all the assets and the index
After this step the df contains the yields
"""
assets = [ticker for ticker in list(df_data.columns.values) if ticker not in ['IR10Y', 'Date']]
for ticker in assets:
    df_data[ticker] = df_data[ticker] / df_data[ticker].shift(1) - 1


"""
Third step : computation of the z's and of the betas and alphas
"""
for ticker in assets:
    df_data[ticker] = df_data[ticker] - df_data['IR10Y']

# Plot some graphs

# plt.figure()
# plt.plot(df_data.index, df_data['SP500'])
# plt.title('Rendements Nets journaliers SP500')
#
# k = 245
# ticker = assets[k]
# plt.figure()
# plt.plot(df_data.index, df_data[ticker])
# plt.title("Net yields of {}".format(ticker))
#
# plt.show()

mu_hat = df_data.mean(axis=0)
mu_hat_star = mu_hat['SP500']
sigma_2_hat = ((df_data['SP500'] - mu_hat_star)**2).mean()


beta_hat = df_data.drop(['SP500', 'IR10Y'], 1)
for ticker in assets:
    if ticker != 'SP500':
        beta_hat[ticker] = (beta_hat[ticker] - mu_hat[ticker])*(df_data['SP500'] - mu_hat_star) / sigma_2_hat

beta_hat = beta_hat.mean(axis=0)


large_stocks = ['AAPL', 'ABT', 'ACN', 'AGN', 'AIG', 'ALL', 'AMGN', 'AMZN', 'APC',
                'AXP', 'BA', 'BAC', 'BIIB', 'BK', 'BLK', 'BMY']

alpha_hat = mu_hat - beta_hat * mu_hat_star

# for ticker in large_stocks:
#     print('Beta of {} :'.format(ticker), beta_hat[ticker])
    # print('Alpha of {} :'.format(ticker), alpha_hat[ticker])

# print(alpha_hat.describe())
# print(alpha_hat.head())
#
# for ticker in assets[100:120]:
#     print('Alpha of {} :'.format(ticker), alpha_hat[ticker])

"""
Step 4 : Statistical tests to know if the alpha's can be considered equal to 0
"""
df = df_data.drop(['Date', 'SP500', 'IR10Y'], 1)

alpha_hat.drop(['SP500', 'IR10Y'], axis=0, inplace=True)
N = alpha_hat.shape[0]
sigma_hat = np.zeros((N,N))

for i in np.arange(N):
    z = np.array(df.iloc[i, :])
    if not np.isnan(z).any():
        v = z - alpha_hat - beta_hat * df_data['SP500'][i]
        sigma_hat += np.outer(v, v)

sigma_hat = sigma_hat / N
sigma_inv = np.linalg.pinv(sigma_hat)

tt = 1. / (1 + mu_hat_star**2 / sigma_2_hat) * N * alpha_hat.transpose().dot(sigma_inv.dot(alpha_hat))
pvalue = sts.t.sf(np.abs(tt), N - 1) * 2
print("Stat :", tt, "\t pvalue :", pvalue)