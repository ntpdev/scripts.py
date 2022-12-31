import time
import locale
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# download historic data from yahoo using selenium and chrome driver
#
#
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def print_ranges(df):
    i = np.argmax(df.close)
    dt = df.index[i]
    print(f'{dt} / {i} high {df.iloc[i].close}')
    i = np.argmin(df.close)
    dt = df.index[i]
    print(f'{dt} / {i} low  {df.iloc[i].close}')

# standardise but use *100 so +1 std is 100
def normaliseAsPerc(v):
    return np.rint(100 * (v - v.mean())/v.std())

def percFromMin(a):
    i = np.argmin(a)
    v = a.iloc[i]
    return np.around((a / v - 1) * 100, 2)

def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['date'], index_col='date', engine='python')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df

# ['Dec 28, 2022', '104.66', '104.89', '101.88', '103.51', '103.51', '418,500']
def parse_yahoo_row(cols):
    d = {}
    # skip Dividend rows
    if not 'D' in cols[1]:
        d['date'] = datetime.strptime(cols[0], '%b %d, %Y').date()
        d['open'] = locale.atof(cols[1])
        d['high'] = locale.atof(cols[2])
        d['low'] = locale.atof(cols[3])
        d['close'] = locale.atof(cols[4])
        d['adjclose'] = locale.atof(cols[5])
        d['volume'] = locale.atof(cols[6])
    return d

def extract_yahoo(soup):
    # search for table
    # <table class="W(100%) M(0)" data-test="historical-prices">
    data = []
    table = soup.find('table', attrs={'data-test':'historical-prices'})

    if table:
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            d = parse_yahoo_row(cols)
            if len(d):
                data.append(d)

    return pd.DataFrame(data)

def parse_page(html, symbol):
    soup = BeautifulSoup(html, 'html.parser')
    df = extract_yahoo(soup)
    print(df)
    fn = f'c:\\users\\niroo\\downloads\\{symbol} {df.iloc[0].date}.csv'
    df.to_csv(fn, index=False)
    print('saved ' + fn)

def download_ticker(driver, symbol):
    driver.get(f'https://finance.yahoo.com/quote/{symbol}/history?p={symbol}')
    time.sleep(1)
    page_down(driver, 5)
    parse_page(driver.page_source, symbol)

def page_down(driver, n):
    e = driver.find_element(By.TAG_NAME, 'body')
    for i in range(n):
        e.send_keys(Keys.PAGE_DOWN)
        time.sleep(.1)

def run_yahoo():
    driver = webdriver.Chrome()
# need to click thru various popups
    driver.get('https://finance.yahoo.com/quote/CELH')
    time.sleep(2)
    buttons = driver.find_element(By.NAME, 'agree').click()
    time.sleep(3)
    driver.find_element(By.CSS_SELECTOR, '[aria-label=Close]').click()
    time.sleep(2)
    driver.find_element(By.LINK_TEXT, 'Historical Data').click()
    time.sleep(2)
    page_down(driver, 5)

    parse_page(driver.page_source, 'CELH')

    download_ticker(driver, 'SPY')
    download_ticker(driver, 'QQQ')
    download_ticker(driver, 'XLK')

#run_yahoo()
df = load_file('c:\\users\\niroo\\downloads\\XLK 2022-12-30.csv')
df['voln'] = normaliseAsPerc(df.volume)
df['perc'] = percFromMin(df.close)
print(df[:19])
print_ranges(df)
