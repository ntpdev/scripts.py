#!/usr/bin/python3
# python -m pip install pymongo
import argparse
from datetime import datetime, date, time, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo 
import platform
import sys
import tsutils as ts
import re
from plotly.offline import init_notebook_mode
from pymongo.mongo_client import MongoClient
import math
from bisect import bisect_right


@dataclass
class MinVolDay:
    startTm: datetime
    startBar: int
    euStartBar: int
    euEndBar: int
    rthStartBar: int
    rthEndBar: int


def samp():
    open_data = [33.0, 33.3, 33.5, 33.0, 34.1]
    high_data = [33.1, 33.3, 33.6, 33.2, 34.8]
    low_data = [32.7, 32.7, 32.8, 32.6, 32.8]
    close_data = [33.0, 32.9, 33.3, 33.1, 33.1]
    dates = [datetime(year=2013, month=10, day=10),
            datetime(year=2013, month=11, day=10),
            datetime(year=2013, month=12, day=10),
            datetime(year=2014, month=1, day=10),
            datetime(year=2014, month=2, day=10)]

    fig = go.Figure(data=[go.Candlestick(x=dates,
                        open=open_data, high=high_data,
                        low=low_data, close=close_data)])

    fig.show()

def samp3LB():
    df = pd.read_csv('d:/3lb.csv', delimiter='\\s+', converters={'date': lambda e: datetime.strptime(e, '%Y-%m-%d')})
    colours = df['dirn'].map({-1: "red", 1: "green"})
    xs = df['date'].dt.strftime('%m-%d')
    fig = go.Figure(data=[go.Bar(x = xs, y = df['close']-df['open'], base = df['open'], marker=dict(color = colours))])
    fig.update_xaxes(type='category')
    fig.show()

def plot_3lb(fname):
    dfall = pd.read_csv(fname, delimiter=',', converters={'date': lambda e: datetime.strptime(e, '%Y-%m-%d')})
    r = len(dfall)-100
    df = dfall.iloc[r:] if r > 0 else dfall
    colours = df['dirn'].map({-1: "red", 1: "green"})
    xs = df['date'].dt.strftime('%m-%d')
    fig = go.Figure(data=[go.Bar(x = xs, y = df['close']-df['open'], base = df['open'], marker=dict(color = colours))])
    fig.update_xaxes(type='category')
    fig.show()

#        color="LightSeaGreen",
def draw_daily_lines(df, fig, tms, idxs):
    for op, cl in idxs:
#        print(f'op {tms.iloc[op]} cl {tms.iloc[cl]}')
        fig.add_vline(x=tms.iloc[op], line_width=1, line_dash="dash", line_color="blue")
        fig.add_vline(x=tms.iloc[cl], line_width=1, line_dash="dash", line_color='grey')
        y = df.Open.iloc[op]
        fig.add_shape(type='line', x0=tms.iloc[op], y0=y, x1=tms.iloc[cl], y1=y, line=dict(color='LightSeaGreen', dash='dot'))

# return a series of bar-index and text labels
def highs(df, window):
    hs = df['High'].rolling(window, center=True).max()
    hsv = df['High'][np.equal(df['High'], hs)]
    t = pd.Series.diff(hsv)
    # remove 0 elements
    return df.High[t[t.ne(0)].index]

def lows(df, window):
    hs = df['Low'].rolling(window, center=True).min()
    hsv = df['Low'][np.equal(df['Low'], hs)]
    # remove adjacent values which are the same by differencing and removing 0
    t = pd.Series.diff(hsv)
    return df.Low[t[t.ne(0)].index]
    

def add_hilo_labels(df, tms, fig):
    hs = highs(df, 21)
    ls = lows(df, 21)

    fig.add_trace(go.Scatter(
        x=[tm for tm in df.loc[hs.index].tm],
        y=hs.add(1),
        text=['%.2f' % p for p in df.loc[hs.index].High],
        mode='text',
        textposition="top center",
        name='local high' ))

    fig.add_trace(go.Scatter(
        x=[tm for tm in df.loc[ls.index].tm],
        y=ls.sub(1),
        text=['%.2f' % p for p in df.loc[ls.index].Low],
        mode='text',
        textposition="bottom center",
        name='local low' ))

def bar_containing(df, dt):
    return (df['Date'] <= dt) & (df['DateCl'] > dt)

# return a high and low range to nearest multiple of n
def make_yrange(df, op, cl, n):
    h = df['High'][op:cl].max() + n // 2
    l = df['Low'][op:cl].min() - n // 2
    return  (l // n)*n, ((h // n) + 1)*n

# pair of start_index:end_index suitable for use with iloc[s:e]
def make_day_index(df):
    # filter by hour > 21 since holidays can have low volume
    is_first_bar = (df['Date'].diff().fillna(pd.Timedelta(hours=1)) > timedelta(minutes=59)) & (df['Date'].dt.hour > 21)
    xs = df[is_first_bar].index.to_list()
    # add index after final bar 
    xs.append(df.shape[0])
    return [ (op,cl) for op,cl in zip(xs, xs[1:]) ]

# return a list of tuples of the op,cl indexes
def make_rth_index(df, day_index):
    is_first_bar = (df['Date'].diff().fillna(pd.Timedelta(hours=1)) > timedelta(minutes=59)) & (df['Date'].dt.hour > 21)
    rth_opens = df[is_first_bar].Date.apply(lambda e: e + np.timedelta64(930 - e.minute, 'm'))
    rth_closes = df[is_first_bar].Date.apply(lambda e: e + np.timedelta64(1320 - e.minute, 'm'))

    # select rows matching time, convert index to a normal col, add col which is date
    ops = df[df.Date.isin(rth_opens)]
    ops2 = ops.reset_index()
    ops2['dt'] = ops2.Date.dt.date

    cls = df[df.Date.isin(rth_closes)]
    cls2 = cls.reset_index()
    cls2['dt'] = cls2.Date.dt.date

    # join dfs on date ie include only days that have open+close
    mrg = pd.merge(ops2, cls2, how='inner', on='dt')
    return [(x,y) for x,y in zip(mrg['index_x'], mrg['index_y'])]

def plot(index):
    df = pd.read_csv(ts.make_filename('es-minvol.csv'), parse_dates=['Date', 'DateCl'], index_col=0)

    # create a string for X labels
    tm = df['Date'].dt.strftime('%d/%m %H:%M')
    fig = color_bars(df, tm, 'strat')
#    fig = go.Figure(data=[go.Candlestick(x=tm, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ES'),
#                        go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])
    xs = make_day_index(df)
    rths = make_rth_index(df, xs)
    draw_daily_lines(df, fig, tm, rths)
    add_hilo_labels(df, tm, fig)

    op, cl = xs[index]
    fig.layout.xaxis.range = [op, cl]
    l, h = make_yrange(df, op, cl, 4)
    fig.layout.yaxis.range = [l, h]
    fig.show()

def color_bars(df, tm, opt):
  if opt == 'strat':
    df['tm'] = tm
    df['btype'] = ts.calc_strat(df)
    dfInside = df.loc[df['btype'] == 0]
    dfUp = df.loc[df['btype'] == 1]
    dfDown = df.loc[df['btype'] == 2]
    dfOutside = df.loc[df['btype'] == 3]
    # find conseq 1 x3 followed by either 2 or 3
    # s = ''.join(str(i) for i in df['btype'].tolist())
    # ms = re.finditer('[01]{3,}[23]+1', s)
    # for m in ms:
    #     print(f'{df['tm'].iloc[m.span()[0]]} {m.group()}')
    # breakpoint()

    fig = go.Figure(data=[go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])
    if 'EMA' in df:
        fig.add_trace(go.Scatter(x=tm, y=df['EMA'], line=dict(color='yellow'), name='ema'))
    fig.add_trace(go.Ohlc(x=dfInside['tm'], open=dfInside['Open'], high=dfInside['High'], low=dfInside['Low'], close=dfInside['Close'], name='ES inside'))
    fig.add_trace(go.Ohlc(x=dfUp['tm'], open=dfUp['Open'], high=dfUp['High'], low=dfUp['Low'], close=dfUp['Close'], name='ES up'))
    fig.add_trace(go.Ohlc(x=dfDown['tm'], open=dfDown['Open'], high=dfDown['High'], low=dfDown['Low'], close=dfDown['Close'], name='ES down'))
    fig.add_trace(go.Ohlc(x=dfOutside['tm'], open=dfOutside['Open'], high=dfOutside['High'], low=dfOutside['Low'], close=dfOutside['Close'], name='ES outside'))
                     
    fig.data[2].increasing.line.color = 'yellow'
    fig.data[2].decreasing.line.color = 'yellow'
    fig.data[3].increasing.line.color = 'green'
    fig.data[3].decreasing.line.color = 'green'
    fig.data[4].increasing.line.color = 'red'
    fig.data[4].decreasing.line.color = 'red'
    fig.data[5].increasing.line.color = 'purple'
    fig.data[5].decreasing.line.color = 'purple'
    return fig
  else:  
    return go.Figure(data=[go.Candlestick(x=tm, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ES'),
                     go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])

def plot_atr():
    df = ts.load_files(ts.make_filename('esh4*.csv'))
    atr = ts.calc_atr(df, 2)
    fig = go.Figure()
    fig.add_trace( go.Scatter(x=atr.index, y=atr, mode='lines', name='ATR5') )
    fig.show()


def plot_tick(days :int):
    '''display the last n days'''
    df = ts.load_files(ts.make_filename('TICK-NYSE*.csv'))
    #  last n days from a dataframe with a datetime index
    idx = ts.day_index(df)
    filtered = df[df.index >= idx.openTime[-days]]
    hi = filtered['High'].quantile(0.95)
    lo = filtered['Low'].quantile(0.05)
    print(f'tick percentiles 5,95  {lo} and {hi}')
    tm = filtered.index.strftime('%d/%m %H:%M')
    fig = px.bar(x=tm, y=filtered.High, base=filtered.Low)
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=tm, y=df['High'], mode='lines', name='high') )
    # fig.add_trace(go.Scatter(x=tm, y=df['Low'], mode='lines', name='low') )
    fig.add_hline(y=hi, line_width=1, line_dash="dash", line_color="grey")
    fig.add_hline(y=lo, line_width=1, line_dash="dash", line_color='grey')
    fig.show()


def create_minVol_index(dfMinVol, day_index):
    # first bar will either be at 23:00 most of the time but 22:00 when US/UK clocks change at different dates
    startTm = dfMinVol.index[0]
    last = dfMinVol.shape[0] - 1
    
    xs = []
    for startTm in day_index.openTime:
        startBar = floor_index(dfMinVol, startTm)
        print(startBar)
        euStart = startTm + timedelta(minutes=600)
        euStartBar = floor_index(dfMinVol, euStart)
        euEndBar, rthStartBar, rthEndBar = -1, -1, -1
        if euStartBar < last:
            euEnd = startTm + timedelta(minutes=929)
            euEndBar = floor_index(dfMinVol, euEnd)
            if euEndBar < last:
                rthStart = startTm + timedelta(minutes=930)
                rthStartBar = floor_index(dfMinVol, rthStart)
                rthEnd = startTm + timedelta(minutes=1320)
                rthEndBar = floor_index(dfMinVol, rthEnd)
        xs.append(MinVolDay(startTm, startBar, euStartBar, euEndBar, rthStartBar, rthEndBar))
    return xs


def plot_mongo(symbol, dt, n):
    df = load_mongo(symbol, dt, n)
    idx = ts.day_index(df)
    dfMinVol = ts.aggregateMinVolume(df, 5000 if idx.shape[0] > 2 else 2500)

    # create a string for X labels
    tm = dfMinVol.index.strftime('%d/%m %H:%M')
    fig = color_bars(dfMinVol, tm, 'strat')
    add_hilo_labels(dfMinVol, tm, fig)
    prevRthHi, prevRthLo = -1, -1
    for i in create_minVol_index(dfMinVol, idx):
        euOpen = dfMinVol.at[dfMinVol.index[i.euStartBar], 'Open']
        fig.add_shape(type='line', x0=tm[i.euStartBar], y0=euOpen, x1=tm[i.euEndBar], y1=euOpen, line=dict(color='LightSeaGreen', dash='dot'))
        if i.rthStartBar > 0:
            fig.add_vline(x=tm[i.rthStartBar], line_width=1, line_dash="dash", line_color="blue")
            if i.rthEndBar > 0:
                fig.add_vline(x=tm[i.rthEndBar], line_width=1, line_dash="dash", line_color="blue")
            rthOpen = dfMinVol.at[dfMinVol.index[i.rthStartBar], 'Open']
            fig.add_shape(type='line', x0=tm[i.rthStartBar], y0=rthOpen, x1=tm[i.rthEndBar], y1=rthOpen, line=dict(color='LightSeaGreen', dash='dot'))
            glbxHi = dfMinVol.High[i.startBar:i.euEndBar].max()
            fig.add_shape(type='line', x0=tm[i.rthStartBar], y0=glbxHi, x1=tm[i.rthEndBar], y1=glbxHi, line=dict(color='Gray', dash='dot'))
            glbxLo = dfMinVol.Low[i.startBar:i.euEndBar].min()
            fig.add_shape(type='line', x0=tm[i.rthStartBar], y0=glbxLo, x1=tm[i.rthEndBar], y1=glbxLo, line=dict(color='Gray', dash='dot'))
            if (prevRthHi > 0):
                fig.add_shape(type='line', x0=tm[i.rthStartBar], y0=prevRthHi, x1=tm[i.rthEndBar], y1=prevRthHi, line=dict(color='chocolate', dash='dot'))
                fig.add_shape(type='line', x0=tm[i.rthStartBar], y0=prevRthLo, x1=tm[i.rthEndBar], y1=prevRthLo, line=dict(color='chocolate', dash='dot'))

            prevRthHi = dfMinVol.High[i.rthStartBar:i.rthEndBar].max()
            prevRthLo = dfMinVol.Low[i.rthStartBar:i.rthEndBar].min()
    fig.show()


def floor_index(df, tm):
    """return index of inverval that includes tm"""
    return df.index.searchsorted(tm, side='right') - 1


def load_mongo(symbol, day, n=1):
    client = MongoClient('localhost', 27017)
    # print(client.admin.command('ping'))
    collection = client['futures'].m1
    dfDays = load_trading_days(collection, symbol, 100000)
    days = dfDays.index.date.tolist()
    t = find_index_range(days, day, n)
    tmStart, tmEnd = map_to_date_range(days, t)
    # tz difference changes normal 23:00 but can be 22:00 so load from 10pm prior day
    print(f'loading {symbol} {tmStart} to {tmEnd}')
    #plot_normvol(dfDays)
    c = collection.find(filter = {'symbol' : symbol, 'timestamp' : {'$gte': tmStart, '$lt': tmEnd}}, sort = ['timestamp'])
    rows = []
    for d in c:
#            pprint.pprint(d)
        r = {
                'Date': d['timestamp'],
                'Open': d['open'],
                'High': d['high'],
                'Low':  d['low'],
                'Close': d['close'],
                'Volume': d['volume'],
                'VWAP': d['vwap']
            }
        rows.append(r)
    df = pd.DataFrame(rows)
    df.set_index('Date', inplace=True)
    df['EMA'] = df.Close.ewm(span=90, adjust=False).mean()
    return df


def load_trading_days(collection, symbol, minVol):
    """return dataframe of complete trading days [date, bar-count, volume, normalised-volume]"""
    cursor = collection.aggregate(
        [{'$match': {'symbol': symbol}},
         {'$group': {
            '_id'   : {'$dateTrunc': {'date': '$timestamp', 'unit': 'day'}}, 
            'count' : {'$sum': 1},
            'volume': {'$sum': '$volume'} } },
         {'$match': {'volume': {'$gte': minVol}}},
         {'$sort': {'_id': 1}}] )
    df = pd.DataFrame(list(cursor))
    v = df.volume
    df['normv'] = (v - v.mean()) / v.std()
    df.set_index('_id', inplace=True)
    df.index.rename('date', inplace=True)
    # print(df)
    return df


def find_index_range(xs, x, n):
    '''given a sorted list xs return start,end index for n elements less than or equal to x. If n is negative x will be the last item.'''
    i = bisect_right(xs, x)
    if i < 1:
        raise ValueError(f'{x} is before {xs[0]}')
    i -= 1
    end = i + int(math.copysign(abs(n)-1, n))
    if n < 0:
        i, end = end, i
    return max(i, 0), min(end, len(xs)-1)


def map_to_date_range(xs, x):
  '''given an array and a tuple of indexes, return a tuple containing the start end datetime.'''
  return datetime.combine(xs[x[0]] - timedelta(days=1), time(22, 0)), datetime.combine(xs[x[1]], time(22, 0))


def parse_isodate(s):
    try:
        return date.fromisoformat(s)
    except ValueError:
        return datetime.now().date()


def plot_normvol(df):
    fig = px.bar(df, x=df.index, y='normv')
    fig.show()

def compare_emas():
    """compare 19 ema on M5 to emas on M1. Nearest is around 90-92"""
    df = ts.load_file('c:\\temp\\ultra\\ESZ3 20231002.csv')
    a = df.Close.resample('5T').first()
    # adjust=False is needed to match usual ema calc
    b = a.ewm(span=19, adjust=False).mean()
    dfm5 = pd.DataFrame({'Close_m5':a, 'ema_m5':b})

    for i in range(79, 99):
        df['ema'] = df.Close.ewm(span=i, adjust=False).mean()
        df2 = df.merge(dfm5, how='inner', left_index=True, right_index=True)
        rmse = ((df2.ema - df2.ema_m5) ** 2).mean() ** 0.5
        print(f'i = {i} rmse={rmse}')


# df.index.indexer_at_time(datetime(2023,10,19,7,15,0))
# df.iloc[495]
# df.loc[datetime(2023,10,19,7,15,0)]
# find exact match
# dfMinVol.loc[datetime(2023,10,18,23,54,0)]
# index of time imm before
# dfMinVol.index.searchsorted(datetime(2023,10,18,23,54,0), side='right') - 1 

parser = argparse.ArgumentParser(description='Plot daily chart')
parser.add_argument('--index', type=int, default=-1, help='Index of day to plot e.g. -1 for last')
parser.add_argument('--tlb', type=str, default='', help='Display three line break [fname]')
parser.add_argument('--mdb', type=str, default='', help='Load from MongoDB [yyyymmdd]')
parser.add_argument('--atr', action='store_true', help='Display ATR')
parser.add_argument('--tick', action='store_true', help='Display tick')
parser.add_argument('--days', type=int, default=1, help='Number of days')

argv = parser.parse_args()
print(argv)
if len(argv.tlb) > 0:
    plot_3lb(argv.tlb)
elif len(argv.mdb) > 0:
    plot_mongo('esh4', parse_isodate(argv.mdb), argv.days)
elif argv.atr:
    plot_atr()
elif argv.tick:
    plot_tick(argv.days)
else:
    plot(argv.index)
#plot_atr()
#hilo(df)
#samp3LB()