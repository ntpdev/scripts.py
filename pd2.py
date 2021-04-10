#!/usr/bin/python3
from datetime import datetime, timedelta
import glob as gb
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def rth_index(df):
    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    idx = df[selector].index.to_series()
    # open will always have a gap before it 23:00 CET to add to get RTH hours 14:30 - 20:59
    idx_open = idx.add(timedelta(hours=15, minutes=30))
    idx_open = df.index.intersection(pd.Index(idx_open))
    idx_close = idx.add(timedelta(hours=21, minutes=59))
    idx_close = df.index.intersection(pd.Index(idx_close))
    return idx_open, idx_close

def rth_bars(df):
    idx_open, idx_close = rth_index(df)
    return aggregate_bars(df, idx_open, idx_close)

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

def load_files(fname):
    dfs = [load_file(e) for e in gb.glob(fname) ]
    return pd.concat(dfs)

def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['Date'], index_col='Date')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df

# return boolean index
def find_price_intersection(df, n, start, end):
    return (df['Low'].iloc[start:end] <= n) & (df['High'].iloc[start:end] >= n)

def calc_hilo(df):
    idx_end = 2600
    hi_count = []
    lo_count = []
    hi_count.append(0)
    lo_count.append(0)
    for i in range(1,idx_end):
        current = df.High.iloc[i]
        ch = 0
        for k in range(i-1, 0, -1):
            prev = df.High.iloc[k]
            if current > prev:
                ch = ch + 1
            elif current < prev:
                break
        
        current = df.Low.iloc[i]
        cl = 0
        for k in range(i-1, 0, -1):
            prev = df.Low.iloc[k]
            if current < prev:
                cl = cl - 1
            elif current > prev:
                break

#        print(f'{i:4d} {ch:4d} {cl:4d} {df.index[i]}  {df.High.iloc[i]:.2f} {df.Low.iloc[i]:.2f}')
        hi_count.append(ch)
        lo_count.append(cl)

    df['HiCount'] = pd.Series(hi_count, index=df.index[0:idx_end], dtype='Int32')
    df['LoCount'] = pd.Series(lo_count, index=df.index[0:idx_end], dtype='Int32')

def make_colour(h,l):
    return 'green' if h > abs(l) else 'red'

print(f'Hello world {datetime.now()}')
#df = load_files('/media/niroo/ULTRA/esh1*')
df = load_files('D:\\esm1*.csv')
df['VWAP'] = calc_vwap(df)
calc_hilo(df)
print('--- RTH bars ---')
df2 = rth_bars(df)
print(df2)


cols = []
idx_end = 2600
c = 'grey'
cols.append(c)
for i in range(1,idx_end):
    h = df.HiCount.iloc[i]
    l = df.LoCount.iloc[i]
    if h > 19:
        c = 'green'
    elif l < -19:
        c = 'red'
    cols.append(c)
#df['Colour'] = pd.Series(cols, index=df.index[0:200], dtype='Int32')
df['Colour'] = pd.Series(cols, index=df.index[0:idx_end])

print(df.iloc[80:120])

fig = go.Figure()
fig.add_trace(go.Bar(x=df.index[0:idx_end],
    y=df.High.iloc[0:idx_end]-df.Low.iloc[0:idx_end],
    base=df.Low.iloc[0:idx_end],
    marker=dict(color=df.Colour.iloc[0:idx_end]) ))
#    color_discrete_map={ '0' : 'blue' }))
#fig.show()