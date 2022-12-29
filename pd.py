#!/usr/bin/python3
from datetime import date, time, timedelta, datetime
import numpy as np
import pandas as pd
import sys
from tsutils import make_filename, load_files, day_index, rth_index, aggregate_daily_bars, calc_vwap, calc_atr, LineBreak

def export_daily(df, fname):
    dt = df.index[-1]
    s = make_filename('%s-%d%02d%02d.csv' % (fname,dt.year, dt.month, dt.day))
    print(f'exporting daily to {s}')
    df.to_csv(s)

def exportNinja(df, outfile):  
    print(f'exporting in Ninja Trader format {outfile} {len(df)}')
    with open(outfile, 'w') as f:
        for i,r in df.iterrows():
            s = '%s;%4.2f;%4.2f;%4.2f;%4.2f;%d\n' % (i.strftime('%Y%m%d %H%M%S'),r['Open'],r['High'],r['Low'],r['Close'],r['Volume'])
            f.write(s)

def exportMinVol(df, outfile):
    df2 = aggregateMinVolume(df, 2500)
    print(f'exporting minVol file {outfile} {len(df2)}')
    df2.to_csv(outfile)

def export_3lb(df, outfile):
    lb = LineBreak(3)
    for i,r in df.iterrows():
        lb.append(r['Close'], i)
    df2 = lb.asDataFrame()
    print(f'exporting 3lb file {outfile} {len(df2)}')
    df2.to_csv(outfile)

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


def print_summary(df):
    di = day_index(df)
    dr = rth_index(di)
    print('--- Daily bars ---')
    df2 = aggregate_daily_bars(df, di)
    export_daily(df2, 'es-daily')
    print(df2)

    print('--- RTH bars ---')
    df2 = aggregate_daily_bars(df, dr)
    export_daily(df2, 'es-daily-rth')
    print(df2)
    export_3lb(df2, make_filename('es-rth-3lb.csv'))


df = load_files(make_filename('esh3*.csv'))
print_summary(df)
df['VWAP'] = calc_vwap(df)
#exportNinja(df, make_filename('ES 09-22.Last.txt'))
exportMinVol(df, make_filename('es-minvol.csv'))
