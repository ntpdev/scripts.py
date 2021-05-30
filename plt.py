#!/usr/bin/python3
import argparse
from datetime import datetime, date, time, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo 
import platform
import sys
from plotly.offline import init_notebook_mode

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

#        color="LightSeaGreen",
def draw_daily_lines(df, fig, tms, idxs):
    for op, cl in idxs:
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

def make_rth_index(df, day_index):
    is_first_bar = (df['Date'].diff().fillna(pd.Timedelta(hours=1)) > timedelta(minutes=59)) & (df['Date'].dt.hour > 21)
    rth_opens = df[is_first_bar].Date.apply(lambda e: e + np.timedelta64(930 - e.minute, 'm'))
    rth_closes = df[is_first_bar].Date.apply(lambda e: e + np.timedelta64(1320 - e.minute, 'm'))
    return [ (op, cl) for op,cl in zip(df[df.Date.isin(rth_opens)].index, df[df.Date.isin(rth_closes)].index) ]


def make_filename(fname):
    p = '/media/niroo/ULTRA/' if platform.system() == 'Linux' else 'd:\\'
    return p + fname

def plot(index):
    df = pd.read_csv(make_filename('cvol22.csv'), parse_dates=['Date', 'DateCl'], index_col=0)

    # create a string for X labels
    tm = df['Date'].dt.strftime('%d/%m %H:%M')
    fig = go.Figure(data=[go.Candlestick(x=tm, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='ES'),
                        go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange'), name='vwap') ])
    xs = make_day_index(df)
    rths = make_rth_index(df, xs)
    draw_daily_lines(df, fig, tm, rths)
    peaks(df, tm, fig)

    op, cl = xs[index]
    fig.layout.xaxis.range = [op, cl]
    l, h = make_yrange(df, op, cl, 4)
    fig.layout.yaxis.range = [l, h]
    fig.show()

parser = argparse.ArgumentParser(description='Plot daily chart')
parser.add_argument('--index', type=int, default=-1, help='Index of day to plot e.g. -1 for last')

argv = parser.parse_args()
plot(argv.index)
#hilo(df)
#samp()