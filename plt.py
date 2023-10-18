#!/usr/bin/python3
# python -m pip install pymongo
import argparse
from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo 
import platform
import sys
import tsutils as ts
from plotly.offline import init_notebook_mode
from pymongo.mongo_client import MongoClient
import pprint

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
    df = pd.read_csv('d:/3lb.csv', delimiter='\s+', converters={'date': lambda e: datetime.strptime(e, '%Y-%m-%d')})
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
    

def peaks(df, tms, fig):
    hs = highs(df, 21)
    ls = lows(df, 21)

#    fig = go.Figure()
#    fig.add_trace(go.Scatter(x=tm, y=df['High'][:600], line=dict(color='orange')))
#    fig.add_trace(go.Scatter(x=tm, y=df['Low'][:600], line=dict(color='cyan')))

#    fig.add_trace(go.Scatter(
#        x=[tm[i] for i in xs[0]],
#        y=[df.High[j] for j in xs[0]],
#        mode='markers',
#        marker=dict(size=8, color='green', symbol='cross' ),
#        name='Detected Peaks' ))

    fig.add_trace(go.Scatter(
        x=[tms[i] for i in hs.index],
        y=hs.add(1),
        text=['%.2f' % y for y in hs],
        mode='text',
        textposition="top center",
        name='local high' ))

    fig.add_trace(go.Scatter(
        x=[tms[i] for i in ls.index],
        y=ls.sub(1),
        text=['%.2f' % y for y in ls],
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
    peaks(df, tm, fig)

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

    fig = go.Figure(data=[go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])
    fig.add_trace(go.Ohlc(x=dfInside['tm'], open=dfInside['Open'], high=dfInside['High'], low=dfInside['Low'], close=dfInside['Close'], name='ES inside'))
    fig.add_trace(go.Ohlc(x=dfUp['tm'], open=dfUp['Open'], high=dfUp['High'], low=dfUp['Low'], close=dfUp['Close'], name='ES up'))
    fig.add_trace(go.Ohlc(x=dfDown['tm'], open=dfDown['Open'], high=dfDown['High'], low=dfDown['Low'], close=dfDown['Close'], name='ES down'))
    fig.add_trace(go.Ohlc(x=dfOutside['tm'], open=dfOutside['Open'], high=dfOutside['High'], low=dfOutside['Low'], close=dfOutside['Close'], name='ES outside'))
                     
    fig.data[1].increasing.line.color = 'yellow'
    fig.data[1].decreasing.line.color = 'yellow'
    fig.data[2].increasing.line.color = 'green'
    fig.data[2].decreasing.line.color = 'green'
    fig.data[3].increasing.line.color = 'red'
    fig.data[3].decreasing.line.color = 'red'
    fig.data[4].increasing.line.color = 'purple'
    fig.data[4].decreasing.line.color = 'purple'
    return fig
  else:  
    return go.Figure(data=[go.Candlestick(x=tm, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ES'),
                     go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])

def plot_atr():
    df = ts.load_files(ts.make_filename('esh2*.csv'))
    atr = ts.calc_atr(df, 2)
    fig = go.Figure()
    fig.add_trace( go.Scatter(x=atr.index, y=atr, mode='lines', name='ATR5') )
    fig.show()

def load_mongo(dt):
    try:
        print('load ' + dt)
        client = MongoClient("localhost", 27017)
        print(client.admin.command('ping'))
        db = client["futures"]
        collection = db.m1
        c = collection.find(filter = {'symbol' : 'esz3', 'timestamp' : {'$gte': datetime(2023, 10, 15, 23, 0)}}, sort = ['timestamp'])
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
        dfMinVol = ts.aggregateMinVolume(df, 2500)
        print(dfMinVol)
        # create a string for X labels
        tm = dfMinVol['Date'].dt.strftime('%d/%m %H:%M')
        fig = color_bars(dfMinVol, tm, '')
        fig.show()
    except Exception as e:
        print(e)

parser = argparse.ArgumentParser(description='Plot daily chart')
parser.add_argument('--index', type=int, default=-1, help='Index of day to plot e.g. -1 for last')
parser.add_argument('--tlb', type=str, default='', help='Display three line break')
parser.add_argument('--mdb', type=str, default='', help='Load from MongoDB [trade-date]')

argv = parser.parse_args()
if len(argv.tlb) > 0:
    plot_3lb(argv.tlb)
elif len(argv.mdb) > 0:
    load_mongo(argv.mdb)
else:
    plot(argv.index)
#plot_atr()
#hilo(df)
#samp3LB()