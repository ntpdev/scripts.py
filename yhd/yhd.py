#!/usr/bin/python3
from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
import glob as gb
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

def calc_pct_change(ser):
    base = ser.iloc[0]
    return round(ser / base - 1, 3) * 100

def make_filename(ticker):
    return f'd:\\{ticker}.csv'

def save(ticker, df):
    fname = make_filename(ticker)
    df.to_csv(fname)
    print('saved ' + fname)

def download(ticker):
    isf = yf.Ticker(ticker)
    df = isf.history(period='2y')
    df2 = df.drop(columns=['Dividends', 'Stock Splits'])
    df2 = df2.round(2)
    df2['PctChg'] = df2.Close.pct_change().round(4) * 100
    df2['HiLo'] = calc_hilo(df2.Close)
    df2['RelChg'] = calc_pct_change(df2.Close)
    return df2

def print_range(df, n):
    h = df.High[-n:].max()
    l = df.Low[-n:].min()
    c = df.Close[-1]
    r = 100*(c-l)/(h-l)
    print(f'{n}d range {l:.2f} {h:.2f} {r:.2f}%')

def print_stats(df):
    xs = df.PctChg.rolling(50).std()
    print(f'50d stddev {xs[-1]:.2f}%')
    print_range(df, 20)
    print_range(df, 50)

# WRKS.L MTRO.L HSW.L
ticker = 'QQQ'
df = download(ticker)
#df = pd.read_csv(make_filename(ticker), parse_dates=['Date'], index_col='Date')
print(df[-49:])
save(ticker, df)
print_stats(df)