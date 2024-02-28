import time
import locale
import math
import pandas as pd
import numpy as np
import re
from datetime import datetime
from bs4 import BeautifulSoup
from glob import glob
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
import tsutils as ts

# pip3 install webdriver-manager
# download historic data from yahoo using selenium and chrome driver
#
#
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def print_ranges(df):
    print('\nrecent range')
    last = df.iloc[-1].close
    i = np.argmax(df.close)
    dt = df.index[i]
    x = df.iloc[i].close
    ago = len(df) - i
    print(f'{dt.date()} / {ago:3d} high {x} {100 * (last / x - 1):6.2f}%')
    i = np.argmin(df.close)
    dt = df.index[i]
    x = df.iloc[i].close
    ago = len(df) - i
    print(f'{dt.date()} / {ago:3d} low  {x} {100 * (last / x - 1):6.2f}%')

def find_initial_swing(s, perc_rev):
    hw = s[0]
    hwi = 0
    lw = s[0]
    lwi = 0
    for i in range(s.size):
        x = s.iat[i]
        if x > hw:
            hw = x
            hwi = i
        elif x < lw:
            lw = x
            lwi = i
        if pdiff(lw, hw, perc_rev):
            return (-1, lwi) if lwi < hwi else (1, hwi)
    return (0, 0)

def find_swings(s, perc_rev):
    dirn, i = find_initial_swing(s, perc_rev)
    xs = []
    if dirn == 0:
        return xs
#    xs.append( {'index': i, 'price': s[i]} )
    hw = s[i]
    hwi = i
    lw = s[i]
    lwi = i
    for i in range(1, s.size):
        x = s.iat[i]
#        print(f'{i} {x} lw {lw} hw {hw} dirn {dirn}')
        if dirn == 1:
            if x > hw:
                # new high in upswing
                hw = x
                hwi = i
                lw = x
                lwi = i
            elif x < lw:
                # new low in upswing
                lw = x
                lwi = i
                if pdiff(hw, lw, perc_rev):
                    #print(f'rev down {i} {x}')
                    xs.append(hwi)
                    dirn = -1
                    hw = x
                    hwi = i
        else:
            if x < lw:
                # new low in down swing
                #print(f'-low {i} {x}')
                lw = x
                lwi = i
                hw = x
                hwi = i
            elif x > hw:
                # new high in down swing
                hw = x
                hwi = i
                if pdiff(lw, hw, perc_rev):
                    #print(f'rev up {i} {x}')
                    xs.append(lwi)
                    lw = x
                    lwi = i
                    dirn = 1
#        print(f'-- loww {lw} lwi {lwi} hiw {hw} hwi {hwi} dirn {dirn}')
# add incomplete swing

    i = s.size - 1
    xs.append(i)
    return xs

def make_swing_table(df, xs):
    # create a new df from some rows - not a view
    ys = df.iloc[xs].loc[:, ('close')]
    zs = pd.concat([ys, ys.pct_change().round(4) * 100], axis=1)
    # rename columns
    zs.columns = ['close', 'swing']
    return zs

# return true if perc diff gt
def pdiff(s, e, p):
    return 100 * abs(e / s - 1) >= p

def test_swings():
    xs = [math.floor(100 + 10 *math.sin(x * .2)) for x in range(0,63)]
    find_swings(pd.Series(xs), 5.0)
    state = UpSwingState(100, 0, [])
    state = state.record(101, 1)


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
        d['adjcl'] = locale.atof(cols[5])
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

    data.reverse()
    return pd.DataFrame(data)

def parse_page(html, symbol):
    soup = BeautifulSoup(html, 'html.parser')
    df = extract_yahoo(soup)
    print(df)
    fn = f'c:\\users\\niroo\\downloads\\{symbol} {df.iloc[-1].date}.csv'
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

def run_yahoo(symbols):
    # check https://chromedriver.chromium.org/downloads if newer version needed and unzip to
    driver = webdriver.Chrome('C:\\Users\\niroo\\AppData\\Local\\Programs\\bin\\chromedriver.exe')
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    wait = WebDriverWait(driver, 10)
# need to click thru various popups
    driver.get('https://finance.yahoo.com/quote/' + symbols[0])
    #time.sleep(2)
    wait.until(EC.element_to_be_clickable((By.NAME, 'agree')))
    buttons = driver.find_element(By.NAME, 'agree').click()
#    time.sleep(4)
#    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, '[aria-label=Close]')))
#    time.sleep(999)
#    driver.find_element(By.CSS_SELECTOR, '[aria-label=Close]').click()
#    driver.find_element(By.XPATH, '//button[text()="Maybe later"]').click()
    time.sleep(2)
    driver.find_element(By.LINK_TEXT, 'Historical Data').click()
    time.sleep(2)
    page_down(driver, 5)

    parse_page(driver.page_source, symbols[0])
    for i in range(1, len(symbols)):
        download_ticker(driver, symbols[i])
#    download_ticker(driver, 'QQQ')
#    download_ticker(driver, 'XLK')

# SPY 2023-01-27.csv
def parse_date(fn):
    m = re.search('\d\d\d\d-\d\d-\d\d', fn)
    return datetime.fromisoformat(m.group(0)).date()

def find_file(symbol):
    files = glob(f'/users/niroo/downloads/{symbol} 20*.csv')
    files.sort(key = lambda e: parse_date(e))
#    print(files)
    return files[-1]

#symbols = ['NVCR', 'SPY', 'QQQ', 'XLK']
symbols = ['SPY', 'QQQ', 'CRWD']
run_yahoo(symbols)
df = load_file(find_file('SPY'))
#load_file('c:\\users\\niroo\\downloads\\SPY 2023-02-03.csv')
df['change'] = pd.Series.diff(df.close)
df['pct_chg'] = pd.Series.pct_change(df.close)
df['voln'] = normaliseAsPerc(df.volume)
df['perc'] = percFromMin(df.close)
df['hilo'] = ts.calc_hilo(df.close)
x = np.diff(df.high)
y = np.diff(df.low)
print(df[-29:])
print_ranges(df)
xs = find_swings(df.close, 5.0)
swings = make_swing_table(df, xs)
print(swings)
#test_swings()
