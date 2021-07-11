#!/usr/bin/python3
from datetime import date, time, timedelta, datetime
import numpy as np
import pandas as pd
import glob as gb
import platform
import sys

def make_filename(fname):
    p = '/media/niroo/ULTRA/' if platform.system() == 'Linux' else 'd:\\'
    return p + fname

def load_files(fname):
    dfs = [load_file(e) for e in gb.glob(fname) ]
    return pd.concat(dfs)

def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['Date'], index_col='Date')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df

def calc_vwap(df):
    is_first_bar = df.index.to_series().diff() != timedelta(minutes=1)
    xs = []
    start = 0
    for i,r in df.iterrows():
        if is_first_bar.loc[i]:
            start = i
        v = np.average( df['WAP'].loc[start:i], weights=df['Volume'].loc[start:i] )
        xs.append(round(v, 2))
    return pd.Series(xs, df.index)

def opening_range(df, start, min_retrace):
    c = 0
    rng_hi = 0
    rng_lo = 9999
    for i,r in df.loc[start:].iterrows():
#        print(r)
        c += 1
        if r.High > rng_hi:
            rng_hi = r.High
        if r.Low < rng_lo:
            rng_lo = r.Low
        if r.Close > rng_lo + min_retrace and r.Close < rng_hi - min_retrace:
            break

    print(f'{c} {rng_lo} {rng_hi}')
    return rng_lo, rng_hi

def orb(df, start, rng_lo, rng_hi):
    c = 0
    state = 0
    dirn = 0
    ti = 0
    open_price = 0
    close_price = 0
    stop_price = 0
    target_price = 0
    for i,r in df.loc[start:].iterrows():
#        print(r)
        c += 1
        if state == 0:
            if r.High > rng_hi:
#                print(f'long {c} {rng_hi} {r.High} {r.Close}')
                ti = i
                open_price = rng_hi + 0.25
                stop_price = open_price - 4
                target_price = open_price + 20
                state = 1
                dirn = 1
            if r.Low < rng_lo:
#                print(f'short {c} {rng_lo} {r.Low} {r.Close}')
                ti = i
                open_price = rng_lo - 0.25
                stop_price = open_price + 4
                target_price = open_price - 10
                state = 1
                dirn = -1
        elif state == 1:
            if dirn == 1 and (r.Close < stop_price or r.Close > target_price):
                state = 2
                close_price = r.Close
                print(f'close long {c} {open_price} {close_price} {close_price - open_price}')
            if dirn == -1 and (r.Close > stop_price or r.Close < target_price):
                state = 2
                close_price = r.Close
                print(f'close short {c} {open_price} {close_price} {close_price - open_price}')
        elif state == 2:
            break
    
    x = {}
    x['Date'] = ti
    x['Direction'] = 'Long' if dirn == 1 else 'Short'
    x['Open'] = open_price
    x['Close'] = close_price
    x['Profit'] = dirn * (close_price - open_price)
    return x

df = load_files(make_filename('esu1*.csv'))
df['VWAP'] = calc_vwap(df)
print(df)

selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
idx = df[selector].index.to_series()
# open will always have a gap before it 23:00 CET to add to get RTH hours 14:30 - 20:59
idx_open = idx.add(timedelta(hours=15, minutes=30))
idx_open = df.index.intersection(pd.Index(idx_open))
print(idx_open)
xs = []
for i in idx_open:
    rng_lo, rng_hi = opening_range(df, i, 2.5)
    x = orb(df, i, rng_lo, rng_hi)
    xs.append(x)

df2 = pd.DataFrame(xs)
print(df2)
print(f"total: {df2['Profit'].sum()}")