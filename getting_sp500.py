import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import re
import pandas as pd
import io

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    tickers.remove("BRK.B")
    tickers.remove("BF.B")

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    # print(tickers)
    return tickers


def get_cookie_and_token(ticker):
    url = 'https://uk.finance.yahoo.com/quote/{}/history'.format(ticker)  # url for a ticker symbol, with a download link
    r = requests.get(url)  # download page

    txt = r.text  # extract html

    cookie = r.cookies['B']  # the cookie we're looking for is named 'B'

    # Now we need to extract the token from html.
    # the string we need looks like this: "CrumbStore":{"crumb":"lQHxbbYOBCq"}
    # regular expressions will do the trick!

    pattern = re.compile('.*"CrumbStore":\{"crumb":"(?P<crumb>[^"]+)"\}')

    for line in txt.splitlines():
        m = pattern.match(line)
        if m is not None:
            crumb = m.groupdict()['crumb']

    return cookie, crumb


def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    # dates of CAPM
    start = dt.datetime(2016, 1, 1)
    end = dt.datetime(2017, 12, 31)

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):

            cookie, crumb = get_cookie_and_token(ticker)
            crumb = crumb.replace('\\u002F', '/')

            data = (ticker, int(start.timestamp()), int(end.timestamp()), crumb)

            url = "https://query1.finance.yahoo.com/v7/finance/download/{0}?period1={1}&period2={2}&interval=" \
                  "1d&events=history&crumb={3}".format(*data)

            data = requests.get(url, cookies={'B': cookie})
            buf = io.StringIO(data.text)  # create a buffer
            # df = pd.read_csv(buf, index_col=0, error_bad_lines=False)
            df = pd.read_csv(buf, index_col=0)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        print(ticker)
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined_closes.csv')

#generates list of tickers for all stocks of the S&P500 and saves it in a pickle file
save_sp500_tickers()

#go fetch data from yahoo finance and store them in CSVs files
get_data_from_yahoo()

#compile all data in one big data frame --- keeps only the Adjusted Close valuer (drops O,H,L,C)
compile_data()

#A few tickers not found in YAHOO finance
# BRK.B, BF.B



