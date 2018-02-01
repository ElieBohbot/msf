import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from datetime import datetime

style.use("ggplot")


df_1 = pd.read_csv('trades_1_simple.csv')
df_2 = pd.read_csv('trades_2_simple.csv')

ax1 = plt.subplot(4, 1, 1)
plt.step(range(len(df_1['price'])), df_1['price'], 'b')
plt.ylabel("price")

ax2 = plt.subplot(412, sharex = ax1)
plt.step(range(len(df_1['volume'])), df_1['volume'], 'k', linewidth=0.8)
plt.xlabel("trades")
plt.ylabel("volume")

ax3 = plt.subplot(4, 1, 3)
plt.step(range(len(df_2['price'])), df_2['price'], 'r')
plt.ylabel("price")

ax4 = plt.subplot(414, sharex = ax1)
plt.step(range(len(df_2['volume'])), df_2['volume'], 'k', linewidth=0.8)
plt.xlabel("trades")
plt.ylabel("volume")

print(df_1.head())
print(df_2.head())

plt.show()


"""
Code used to generate the simple csv's
"""
# df_1 = pd.read_csv('trades_1.csv', sep='\t')
# df_2 = pd.read_csv('trades_2.csv', sep='\t')
#
# df_1_simple = df_1[['Time', 'price', 'volume']]
# df_2_simple = df_2[['Time', 'price', 'volume']]
#
# print(df_1_simple.head())
# print(list(df_1_simple.columns.values))
#
#
# df_1_simple.rename(columns={'Time': 'time'}, inplace=True)
# df_2_simple.rename(columns={'Time': 'time'}, inplace=True)
#
#
# def epoch_to_date(epoch):
#     dt = datetime.fromtimestamp(epoch // 1000000000)
#     s = dt.strftime('%H:%M:%S')
#     s += '.' + str(int(epoch % 1000000000)).zfill(9)
#     return s
#
# df_1_simple.loc[:,'hour'] = np.array([epoch_to_date(epoch) for epoch in df_1_simple['time']])
# df_2_simple.loc[:,'hour'] = np.array([epoch_to_date(epoch) for epoch in df_2_simple['time']])
#
# df_1_simple.to_csv('trades_1_simple.csv')
# df_2_simple.to_csv('trades_2_simple.csv')
