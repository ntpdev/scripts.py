#!/usr/bin/python3
from datetime import date, time, timedelta, datetime
import numpy as np
import pandas as pd
import glob as gb
import platform

# import tsutils as ts
# df = ts.load_file(ts.make_filename('esu1 20210705.csv'))

# Time series utility functions

# create a df [date, openTime, closeTime] for each trading day
def day_index(df):
    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    idx = df[selector].index.array
    idx_close = df[selector.shift(periods=-1, fill_value=True)].index.array
    return pd.DataFrame({'openTime':idx, 'closeTime':idx_close}, index=idx_close.date)


# create a df [date, openTime, closeTime] for each rth day from an index of trading days
def rth_index(day_index):
    fulldays = day_index[day_index.closeTime - day_index.openTime == timedelta(hours=22, minutes=59)]
    rth_open = fulldays.openTime + timedelta(hours=15, minutes=30)
    rth_close = fulldays.openTime + timedelta(hours=21, minutes=59)
    return pd.DataFrame( {'openTime':rth_open, 'closeTime':rth_close }, index=fulldays.index )


# create a new DF which aggregates bars using a daily index
def aggregate_daily_bars(df, daily_index):
    rows = []
    for i,r in daily_index.iterrows():
        rows.append(aggregate(df, r['openTime'], r['closeTime']))

    daily = pd.DataFrame(rows, index=daily_index.index)
    daily['Change'] = daily['Close'].sub(daily['Close'].shift())
    daily['DayChg'] = daily['Close'].sub(daily['Open'])
    daily['Range'] = daily['High'].sub(daily['Low'])
    return daily


# return a row which aggregates bars between inclusive indexes
def aggregate(df, s, e):
    r = {}
    r['Open'] = df.Open[s]
    r['High'] = df.High[s:e].max()
    r['Low'] = df.Low[s:e].min()
    r['Close'] = df.Close[e]
    r['Volume'] = df.Volume[s:e].sum()
    vwap = np.average( df.WAP[s:e], weights=df.Volume[s:e] )
    r['VWAP'] = round(vwap, 2)
    return r


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