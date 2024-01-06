#!/usr/bin/env python3
# Note chmod +x *.py
# ensure Unix style line endings

from datetime import date, datetime, timedelta
from twelvedata import TDClient
import requests as req
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as subp
import tsutils
from pathlib import Path
import os
from fnmatch import fnmatch

# pip install twelvedata

APIKEY = 'dc33458b019d4bf59513212a49f3907d'

td = TDClient(apikey=APIKEY)

def make_fullpath(fn: str) -> Path:
    return Path.home() / 'Downloads' / fn


def make_filename(symbol: str, dt: date) -> Path:
    return make_fullpath(f'{symbol} {dt.isoformat()}.csv')


def list_cached_files(symbol: str):
    p = Path.home() / 'Downloads'
    spec = symbol + '*.csv'
    xs = [f for f in os.listdir(p) if fnmatch(f, spec)]
    xs.sort(reverse=True)
    return xs


def load_file(fname: str):
    df = pd.read_csv(make_fullpath(fname), parse_dates=['datetime'], index_col='datetime', engine='python')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df


def json_to_df(objs):
    return pd.DataFrame(objs['values'])


def plot(symbol, df):
    pts = -500
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[pts:], y=df[pts:]['close'], mode='lines', name=symbol))
    fig.add_trace(go.Scatter(x=df.index[pts:], y=df[pts:]['ema19'], mode='lines', name='ema19'))
    # fig1 = px.line(df[pts:], x=df.index[pts:], y=['close', 'ema19'], title=symbol, template='plotly_dark')
    x = df[pts:]
    x = x[x['hilo'] > 19]
    # print(x)
    fig.add_trace(go.Scatter(x=x.index, y=x['close'], mode='markers', name='20d high', marker=dict(color='green')))
    x = df[pts:]
    x = x[x['hilo'] < -19]
    fig.add_trace(go.Scatter(x=x.index, y=x['close'], mode='markers', name='20d low', marker=dict(color='red')))
    # fig2 = px.scatter(x, x=x.index, y='close', color="green")

    # fig3 = px.Figure(data=fig1.data + fig2.data)
    # fig3.show()
    fig.show()


def plot_cumulative(df):
    # Create a barchart using the 'perc' column from the same dataframe
    fig = subp.make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df['dtOpen'], y=df['perc'], name='perc'))
    fig.add_trace(go.Scatter(x=df['dtOpen'], y=df['cumulative'], mode='lines'), secondary_y=True)
    # Show the final figure
    fig.show()


def plot_heatmap(df):
    d = []
    for i,r in df.iterrows():
        s = r['system'].split(',')
        entry = s[0]
        ex = s[1]
        v = r['cumulative']
        # v = r['maxdd']
        d.append({'entry':entry, 'exit':ex, 'value':v})
    map = pd.DataFrame(d)
    # breakpoint()
    # Add a heatmap trace to the figure
    fig = go.Figure(
        data=go.Heatmap(
        x=map['entry'],
        y=map['exit'],
        z=map['value'],
        colorscale='Viridis',
        showscale=True))

    # Set the title and axis labels
    # fig.update_layout(
    #     title='Heatmap',
    #     xaxis_title='Entry',
    #     yaxis_title='Exit',
    #     zaxis_title='Value'
    # )

    # Show the plot
    fig.show()


# standardise but use *100 so +1 std is 100
def normaliseAsPerc(v):
    return np.rint(100 * (v - v.mean())/v.std())


# returns {'datetime': '1993-01-29', 'unix_time': 728317800} for SPY
def load_earliest_date(symbol):
    s = f'https://api.twelvedata.com/earliest_timestamp?symbol={symbol}&interval=1day&apikey={APIKEY}'
    response = req.get(url=s)
    if response.status_code == 200:
        objs = response.json()
        print(objs)


def load_twelve_data(symbol, days=250):
    print(f'loading {symbol}')
    ts = td.time_series(symbol=symbol, interval='1day', outputsize=days, dp=2, order='ASC')
    df = ts.with_ma(ma_type='SMA', time_period=150).with_ma(ma_type='SMA', time_period=50).with_ma(ma_type='EMA', time_period=19).as_pandas()

    df.rename(columns={'ma1':'sma150', 'ma2':'sma50', 'ma3':'ema19'}, inplace=True)
    print(df.tail())
    fname = make_filename(symbol, df.index[-1].date())
    df['change'] = pd.Series.diff(df.close).round(2)
    df['pct_chg'] = (pd.Series.pct_change(df.close) * 100).round(2)
    df['voln'] = normaliseAsPerc(df.volume)
    # df['perc'] = percFromMin(df.close)
    df['hilo'] = tsutils.calc_hilo(df.close)
    df.to_csv(fname)
    print(f'saved {symbol} {df.index[0].date()} to {df.index[-1].date()} shape={df.shape}')
    print(fname)
    return df


def load_twelve_data_raw(symbols):
    dt = (datetime.today().date() - timedelta(days=365)).isoformat()
    url = f'https://api.twelvedata.com/time_series?apikey={APIKEY}&interval=1day&start_date={dt}&symbol={symbols}&type=etf&format=JSON&dp=2&order=ASC'
    print(url)
    response = req.get(url=url)
    if response.status_code == 200:
        objs = response.json()
        df = json_to_df(objs)
        fn = f'c:\\users\\niroo\\downloads\\{symbols} {dt.date()}.csv'
        df.set_index('datetime', inplace=True)
        df.to_csv(fn)
        print('saved ' + fn)


def scan(df, entryHi, exitLo, stopPerc):
    xs = []
    state = 0
    stop = None
    targetPerc = 1.1
    for i, row in df.iterrows():
        if state == 0 and row['hilo'] > entryHi:
            entry = i
            state = 1
            stop = row['close'] * stopPerc
            target = row['close'] * targetPerc
        elif state == 1 and (row['close'] < stop or row['hilo'] < exitLo):
        # elif state == 1 and row['hilo'] < -19:
           xs.append((entry, i))
           state = 0
        elif state == 1 and row['close'] > target:
            xs.append((entry, i))
            state = 0

    ys = []
    for ent, ex in xs:
        d = {'dtOpen': ent,
             'dtClose': ex,
             'open': df.at[ent, 'close'],
             'close': df.at[ex, 'close']}
        ys.append(d)
    df2 = pd.DataFrame(ys)
    df2['points'] = df2['close'] - df2['open']
    df2['perc'] = (df2['points'] / df2['open']) * 100.
    df2['cumulative'] = (df2['close'] / df2['open']).cumprod()
    df2['drawdown'] = df2['cumulative'] - df2['cumulative'].expanding().max() 
    # print(df2)

    # if entryHi == 15 and exitLo == -30:
    #     plot_cumulative(df2)

    pts = df2['points']
    wins = pts > 0
    losses = pts < 0
    return {'system': f'{entryHi},{exitLo},{stopPerc}',
            'pts': pts.sum(),
            'cumulative': df2['cumulative'].iat[-1],
            'maxdd': round(df2['drawdown'].min() * 100.,2),
            'winC': pts[wins].count(),
            'winT': pts[wins].sum(),
            'lossC':pts[losses].count(),
            'lossT':pts[losses].sum()
            }


def plot_swings(df):
    #        fig = px.line(x=swings.index, y=swings['close'])
    fig = px.bar(x=df.index, y=df['change'])
    fig.update_layout(xaxis_type='category') # treat datetime as category
    fig.show()


def main():
    df = load_twelve_data('tlt', 510)
    xs = list_cached_files('tlt')
    if len(xs) > 0:
        df = load_file(xs[0])
        #df = load_file('c:\\users\\niroo\\downloads\\spy-lt.csv')
        swings = tsutils.find_swings(df['close'], 5.0)
        print(df[-20:])
        print('\n-- swings')
        print(swings)
        # print drawdowns if in upswing
        r = swings.iloc[-1]
        if r.change > 0:
            print(f'\n--- p/b limits\nhigh {r.end}\n2%   {r.end * .98:.2f}')
            print(f'5%   {r.end * .95:.2f}\n10%  {r.end * .9:.2f}')
        #plot_swings(swings)
        print(swings[swings.change < 0].change.describe())

def main_concat():
    xs = list_cached_files('spy')
    if len(xs) > 0:
        df = load_file('c:\\users\\niroo\\downloads\\spy 2023-12-15.csv')
        df2 = load_file(xs[0])
        df_updated = pd.concat([df, df2[-30:]])
        df_updated.drop_duplicates(inplace=True)
        df_updated.to_csv(make_fullpath('spy-lt.csv'))
        print(df_updated)


if __name__ == '__main__':
    main()
    #load_earliest_date('spy')
    #df = load_file('c:\\users\\niroo\\downloads\\spy 2023-12-15.csv')
    #plot('spy', df)
    #scan(df, 19, -35, .975)
    # xs = []
    # for x in range(10, 25):
    #     for y in range(19,50):
    #         d = scan(df, x, -y, .975)
    #         xs.append(d)
    # df2 = pd.DataFrame(xs)
    # print(df2)
    # plot_heatmap(df2)
