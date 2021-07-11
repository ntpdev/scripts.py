#!/usr/bin/python3
from datetime import date, time, timedelta, datetime
import numpy as np
import pandas as pd
import glob as gb
import platform
import sys

def exportNinja(df, outfile):
    print(df.tail())
    with open(outfile, 'w') as f:
        for i,r in df.iterrows():
            s = '%s;%4.2f;%4.2f;%4.2f;%4.2f;%d\n' % (i.strftime('%Y%m%d %H%M%S'),r['Open'],r['High'],r['Low'],r['Close'],r['Volume'])
            f.write(s)

def daily_bars(df):
    # find start & end indexes by looking for gaps in index
    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    return aggregate_bars(df, df[selector].index, df[selector.shift(periods=-1, fill_value=True)].index)

def rth_bars(df):
    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    idx = df[selector].index.to_series()
    # open will always have a gap before it 23:00 CET to add to get RTH hours 14:30 - 20:59
    idx_open = idx.add(timedelta(hours=15, minutes=30))
    idx_open = df.index.intersection(pd.Index(idx_open))
    idx_close = idx.add(timedelta(hours=21, minutes=59))
    idx_close = df.index.intersection(pd.Index(idx_close))
    return aggregate_bars(df, idx_open, idx_close)

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

# create a new DF which aggregates bars between inclusive indexes
def aggregate_bars(df, idxs_start, idxs_end):
    rows = []
    dts = []
    for s,e in zip(idxs_start, idxs_end):
        dts.append(e.date())
        r = {}
        r['Open'] = df.Open[s]
        r['High'] = df.High[s:e].max()
        r['Low'] = df.Low[s:e].min()
        r['Close'] = df.Close[e]
        r['Volume'] = df.Volume[s:e].sum()
        vwap = np.average( df.WAP[s:e], weights=df.Volume[s:e] )
        r['VWAP'] = round(vwap, 2)
        rows.append(r)
    daily = pd.DataFrame(rows, index=dts)
    daily['Change'] = daily['Close'].sub(daily['Close'].shift())
    daily['DayChg'] = daily['Close'].sub(daily['Open'])
    daily['Range'] = daily['High'].sub(daily['Low'])
    return daily

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
#    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    selector = df.index.to_series().diff() != timedelta(minutes=1)
    openbar = (df.index.minute == 0) & selector
    lastbar = selector.shift(-1, fill_value=True)
    eur_open = date(2021,1,1)
    rth_open = date(2021,1,1)
    for i,r in df.iterrows():
        if openbar.loc[i]:
            eur_open = i + timedelta(hours=8, minutes=59)
            rth_open = i + timedelta(hours=15, minutes=29)
        acc = single(i,r,1) if len(acc) == 0 else combine(acc, i, r, 1)
        #if i == datetime(2021,4,20,14,29,0):
        #    breakpoint()
        if acc['Volume'] >= minvol or lastbar.loc[i] or i == eur_open or i == rth_open:
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

def print_summary(df):
    print('--- Daily bars ---')
    df2 = daily_bars(df)
    print(df2)

    print('--- RTH bars ---')
    df2 = rth_bars(df)
    print(df2)

#df = pd.read_csv("../../Downloads/spy.csv")
#print(df.tail())

# find bar at certain time
# df[df['Date'].apply(lambda e:e.time() == dt.time(14,30) )]
# i = df[df['Date'].apply(lambda e:e.time() == dt.time(14,30) )].index
# df.iloc[i]
df = load_files(make_filename('esu1*.csv'))
print_summary(df)
df['VWAP'] = calc_vwap(df)
print(df)
exportNinja(df, 'd:\\ES 09-21.Last.txt')
#hilo(df, 4)

#df.to_csv('d:\z.csv')
df2 = aggregateMinVolume(df, 2500)
df2.to_csv(make_filename('cvol22.csv'))
