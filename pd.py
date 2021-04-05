#!/usr/bin/python
from datetime import date, time, timedelta
import numpy as np
import pandas as pd
import glob as gb
import sys

def exportNinja(df, outfile):
#    df = pd.read_csv(infile, parse_dates=['Date'], index_col=0)
    print(df.tail())
    with open(outfile, 'w') as f:
        for i,r in df.iterrows():
            s = '%s;%4.2f;%4.2f;%4.2f;%4.2f;%d\n' % (i.strftime('%Y%m%d %H%M%S'),r['Open'],r['High'],r['Low'],r['Close'],r['Volume'])
            f.write(s)

def calc_vwap(df):
    xs = []
    start = 0
    for i,r in df.iterrows():
        if (isfirstbar(i)):
            start = i
        x = np.average( df['WAP'].loc[start:i], weights=df['Volume'].loc[start:i] )
        xs.append(round(x, 2))
    return pd.Series(xs, df.index)

def aggregrate_bars_between(df, tm_open, tm_close):
    rows = []
    # find indexes of open & close bars
    ops = df.at_time( tm_open ).index
    cls = df.at_time( tm_close ).index
    for op,cl in zip(ops, cls):
        # slicing a dataframe by index uses an inclusive range
        acc = aggregate(df.loc[op:cl])
        rows.append(acc)
    return pd.DataFrame(rows)

# inclusive end
def aggregate(df):
    acc = {}
    for i,r in df.iterrows():
        acc = single(i,r,1) if len(acc) == 0 else combine(acc, i, r, 1)
    return acc

def aggregateMinVolume(df, minvol):
    rows = []
    acc = {}
    for i,r in df.iterrows():
        acc = single(i,r,1) if len(acc) == 0 else combine(acc, i, r, 1)
        if acc['Volume'] >= minvol or islastbar(i) :
            rows.append(acc)
            acc = {}
    if len(acc) > 0:
        rows.append(acc)
    return pd.DataFrame(rows)

def single(dt_fst, fst, period):
    r = {}
    r['Date'] = dt_fst
    r['DateCl'] = dt_fst + timedelta(minutes=period)
    r['Open'] = fst['Open']
    r['High'] = fst['High']
    r['Low'] = fst['Low']
    r['Close'] = fst['Close']
    r['Volume'] = fst['Volume']
    r['VWAP'] = fst['VWAP']
    return r

def combine(acc, dt_snd, snd, period):
    r = {}
    r['Date'] = acc['Date']
    r['DateCl'] = dt_snd + timedelta(minutes=period)
    r['Open'] = acc['Open']
    r['High'] = max(acc['High'], snd['High'])
    r['Low'] = min(acc['Low'], snd['Low'])
    r['Close'] = snd['Close']
    r['Volume'] = acc['Volume'] + snd['Volume']
    r['VWAP'] = snd['VWAP']
    return r

def isRTH(d):
    op = time(13,30)
    cl = time(20,15)
    return d.time() >= op and d.time() < cl

def islastbar(d):
    return (d.hour == 20 and (d.minute == 14 or d.minute == 59)) or (d.hour == 13 and d.minute == 29)

def isfirstbar(d):
    return (d.hour == 22 and d.minute == 00) or (d.hour == 13 and d.minute == 30)

def hilo(df, rev):
    xs = []
    hw = df.High.iloc[0]
    hwp = 0
    lw = df.Low.iloc[0]
    lwp = 0
    hf = False
    lf = False
    c = 0
    for i,r in df.iterrows():
        if r.High > hw:
            hw = r.High
            hwp = i
            hf = True
        elif hw - r.High > rev and hf:
            print('High %.2f' % df.High.loc[hwp])
#            hf = False
            lf = False
            lw = r.Low
            lwp = i

        if r.Low < lw:
            lw = r.Low
            lwp = i
            lf = True
        elif r.Low - lw > rev and lf:
#           print('Low  %.2f' % df.Low.loc[lwp:i])
            print(df.loc[lwp:i])
#            lf = False
            hf = False
            hw = r.High
            hwp = i
            print(df.iloc[50:70])
            c =  c + 1
            if c == 2:
                sys.exit(0)

def aggregate_daily_bars(df):
    df3 = aggregrate_bars_between(df, time(13,30) ,time(20,14))
    df3['Dt'] = df3['Date'].apply(lambda e: e.date())
    df3.set_index('Dt', inplace=True)
    df3['Volume'] = df3['Volume'].astype('int64')
# copy to new DF
    daily = df3[['Open', 'High', 'Low', 'Close', 'VWAP', 'Volume']].copy()
    daily.index.rename('Date', inplace=True)
    daily['Change'] = df3['Close'].sub(df3['Close'].shift())
    daily['DayChg'] = df3['Close'].sub(df3['Open'])
    daily['Range'] = df3['High'].sub(df3['Low'])
    return daily

# https://firstratedata.com/i/futures/ES
def fn1():
    column_names = ["TimeStamp", "open", "high", "low", "close", "volume"]
    df = pd.read_csv("d:\esz19.txt",
        names=column_names,
        parse_dates=["TimeStamp"],
        index_col=["TimeStamp"] )
    dfd = df.resample("1H").agg({'open': 'first', 'close': 'last', 'high' : 'max', 'low' : 'min', 'volume': 'sum'})
    dfd = dfd[dfd.volume > 1000]
    print(dfd.tail(19))

def load_files(contract):
    dfs = [load_file(e) for e in gb.glob( f'd:\\{contract}*.csv') ]
    return pd.concat(dfs)

def load_file(fn):
    df = pd.read_csv(fn, parse_dates=['Date'], index_col='Date')
    print('loaded %s %d x %d' % (fn, df.shape[0], df.shape[1]) )
    return df

#df = pd.read_csv("../../Downloads/spy.csv")
#print(df.tail())

# find bar at certain time
# df[df['Date'].apply(lambda e:e.time() == dt.time(14,30) )]
# i = df[df['Date'].apply(lambda e:e.time() == dt.time(14,30) )].index
# df.iloc[i]
#df = pd.read_csv('d:\esm1 20210322.csv', parse_dates=['Date'], index_col='Date')
df = load_files('esm1')
df['VWAP'] = calc_vwap(df)
print(df.head())
#exportNinja(df, 'd:\ESM1.Last.txt')
#hilo(df, 4)

daily = aggregate_daily_bars(df)
print(daily)

#df.to_csv('d:\z.csv')
df2 = aggregateMinVolume(df, 2500)
print(df2.head())
df2.to_csv('d:\cvol22.csv')

