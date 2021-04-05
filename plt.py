#!/usr/bin/python3
from datetime import datetime, date, time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.offline as pyo 
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


def drawVert(fig, tms):
    for tm in tms:
        if tm.endswith("13:30"):
            fig.add_vline(x=tm, line_width=1, line_dash="dash", line_color="blue")
        if tm.endswith("19:59"):
            fig.add_vline(x=tm, line_width=1, line_dash="dash", line_color="black")


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

# return a tuple of (open_index, close_index, high_bound, lower_bound)
def find_day_index(df, n):
    xs = []
    op = df[df['Date'].dt.time == time(22,00)].index
    cl = df[df['Date'].dt.time == time(19,59)].index
    for x,y in zip(op, cl):
        h = df['High'][x:y+1].max()
        l = df['Low'][x:y+1].min()
        xs.append( (x,y, ((h // n) + 1)*n  , ((l // n)*n)) )
    return xs

#    fig.show()


df = pd.read_csv('d:\\cvol22.csv', parse_dates=['Date', 'DateCl'], index_col=0)
print(df.tail())
# create a string for X labels
tm = df['Date'].dt.strftime('%d %H:%M')
fig = go.Figure(data=[go.Candlestick(x=tm, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']),
    go.Scatter(x=tm, y=df['VWAP'], line=dict(color='orange')) ])
drawVert(fig, tm)
peaks(df, tm, fig)
#xs = find_day_index(df, 5)
#x = xs[-1]
#fig.layout.xaxis.range = [x[0], x[1]]
#fig.layout.yaxis.range = [x[3], x[2]]
fig.show()

#hilo(df)
#samp()