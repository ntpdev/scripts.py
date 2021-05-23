#!/usr/bin/python3
from datetime import datetime, date, time, timedelta
import argparse
import numpy as np
import pandas as pd
import glob as gb
import platform
import sys
import yfinance as yf

def count_back(xs, i):
    current = xs.iloc[i]
    c = 0
    for k in range(i-1, -1, -1):
        prev = xs.iloc[k]
        if c > 0:
            if current >= prev:
                c += 1
            else:
                break
        elif c < 0:
            if current <= prev:
                c -= 1
            else:
                break
        else:
            c = 1 if current >= prev else -1

    return c

def calc_hilo(ser):
    cs = []
    cs.append(0)
    for i in range(1, ser.size):
        cs.append(count_back(ser, i))
    return pd.Series(cs, ser.index)

def calc_rel_change(ser):
    base = ser.iloc[0]
    return round(ser / base - 1, 3) * 100

def calc_extended_run(df, n):
    df['SMA'] = df.Close.rolling(n).mean()
    df['Diff'] = df.Close - df.SMA
    df['Q'] = np.sign(df.Diff)
    ys = df.Q
    df['Run'] = ys * (ys.groupby((ys != ys.shift()).cumsum()).cumcount()+1)
    return df

def find_pullbacks(df):
    in_run = False
    in_trade = False
    ds = []
    ixs = []
    ix = None
    d = {}
    for i,r in df.iterrows():
        if r.Run > 4:
            in_run = True
        if in_run and r.Run <= -1:
            d['Open'] = r.Close
            in_run = False
            in_trade = True
            ix = i
        if in_trade and r.Run >= 1:
            d['Close'] = r.Close
            ds.append(d)
            ixs.append(ix)
            d = {}
            in_trade = False
    trades = pd.DataFrame(ds, index=ixs)
    trades['Points'] = ((trades.Close - trades.Open) / trades.Open) * 100
    trades['Total'] = trades.Points.cumsum()
    return trades

def make_filename(fname):
    p = '/media/niroo/ULTRA/' if platform.system() == 'Linux' else 'd:\\'
    return f'{p}{fname}.csv'

def download_save(ticker):
    df = download(ticker)
    save(ticker, df)
    return df

def save(ticker, df):
    fname = make_filename(ticker)
    df.to_csv(fname)
    print('saved ' + fname)

def download(ticker):
    t = yf.Ticker(ticker)
    df = t.history(period='2y')
    print('download...' + ticker)
    df2 = df.drop(columns=['Dividends', 'Stock Splits'])
    df2 = df2.round(2)
    df2['PctChg'] = df2.Close.pct_change().round(4) * 100
    df2['HiLo'] = calc_hilo(df2.Close)
    df2['RelChg'] = calc_rel_change(df2.Close)
    return df2

def print_range(df, n):
    h = df.High[-n:].max()
    l = df.Low[-n:].min()
    c = df.Close[-1]
    r = 100*(c-l)/(h-l)
    print(f'{n}d range {l:.2f} {h:.1f} {r:.1f}%')

def print_stats(df, n):
    print(df[-49:])
    xs = df.PctChg.rolling(n).std()
    print(f'{n}d stddev {xs[-1]:.2f}%')
    print_range(df, 20)
    print_range(df, 50)

def process(ticker):
# WRKS.L MTRO.L HSW.L VDTK.L
    df = download_save(ticker)
    #df = pd.read_csv(make_filename(ticker), parse_dates=['Date'], index_col='Date')
    df = calc_extended_run(df, 5)
    print(df[-49:])
    print(find_pullbacks(df))
    #save(ticker, df)
    print_stats(df, 50)

parser = argparse.ArgumentParser(description='Download historic prices from yahoo')
parser.add_argument('tickers', metavar='Ticker', nargs='+', help='tickers eg SPY ISF.L')

args = parser.parse_args()
for t in args.tickers:
    process(t)
