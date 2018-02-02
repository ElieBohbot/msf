import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

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


"""
Second step : computing the yields for all the assets and the index
"""
assets = [ticker for ticker in list(df_data.columns.values) if ticker not in ['IR10Y', 'Date']]
for ticker in assets:
    df_data[ticker] = df_data[ticker] / df_data[ticker].shift(1) - 1

print(df_data.describe())

k = 300
ticker = assets[k]
plt.plot(df_data.index, df_data[ticker])
plt.title("Yields of {}".format(ticker))
plt.show()




# data = df_data.as_matrix()
# num_dates_data = data.shape[0]
# num_assets = data.shape[1]
"""
Here data is the matrix of correct values for s&p500
504 date (daily)
495 assets
"""
#%%
# data = data.transpose()
# num_dates = num_dates_data - 1
# y = np.zeros([num_assets, num_dates])
# for i in range(num_dates):
#     y[:, i] = data[:, i+1] / data[:, i] - 1
# #%%
# k = 300
# plt.plot(y[k, :])
# plt.title(asset[k])









