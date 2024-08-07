#!/usr/bin/python3
from datetime import datetime, timedelta, time
import glob as gb
import pandas as pd
import numpy as np
from io import StringIO
#from pathlib import Path
from tsutils import aggregate, aggregateMinVolume, make_filename, load_files, load_file, day_index, day_index2, rth_index, aggregate_daily_bars, calc_vwap, calc_atr, save_df
import plotly.graph_objects as go
from rich.console import Console

console = Console()

def test_tick():
    # load the string tick into a pandas dataframe. make the column Date the index
    tick = """
,Date,Open,High,Low,Close,Volume,WAP,BarCount
0,20240724 14:30:00,-71.00,-63.00,-520.00,-402.00,0,0.000,31
1,20240724 14:31:00,-402.00,-340.00,-491.00,-379.00,0,0.000,31
2,20240724 14:32:00,-379.00,-303.00,-432.00,-380.00,0,0.000,31
3,20240724 14:33:00,-380.00,-195.00,-398.00,-215.00,0,0.000,31
4,20240724 14:34:00,-215.00,-48.00,-282.00,-48.00,0,0.000,31
5,20240724 14:35:00,-48.00,41.00,-169.00,-10.00,0,0.000,31
    """

    tick_df = pd.read_csv(StringIO(tick), index_col='Date', parse_dates=['Date'])
    tick_df.drop(columns=[tick_df.columns[0], 'Volume', 'WAP', 'BarCount'], inplace=True)
    tick_df.columns = tick_df.columns.str.lower()

    # load the string futures into a pandas dataframe. make the column Date the index.
    futures = """
,Date,Open,High,Low,Close,Volume,WAP,BarCount
928,20240724 14:28:00,5405.00,5411.50,5404.25,5407.00,5314,5408.550,1515
929,20240724 14:29:00,5407.00,5407.75,5402.75,5404.25,4080,5405.425,1320
930,20240724 14:30:00,5404.50,5405.50,5398.00,5403.25,16809,5401.400,6437
931,20240724 14:31:00,5403.25,5404.25,5394.00,5398.50,10444,5397.850,4031
932,20240724 14:32:00,5398.50,5401.75,5396.50,5398.25,9685,5398.875,3545
933,20240724 14:33:00,5398.25,5400.75,5391.00,5391.25,9019,5394.875,3408
934,20240724 14:34:00,5391.25,5392.00,5386.75,5387.00,10123,5389.325,3358
935,20240724 14:35:00,5387.00,5393.25,5387.00,5389.00,11340,5390.125,4120
936,20240724 14:36:00,5389.25,5398.00,5387.50,5396.00,10767,5393.700,3706
937,20240724 14:37:00,5396.25,5401.00,5394.50,5397.00,11314,5397.475,3958
    """

    futures_df = pd.read_csv(StringIO(futures), index_col='Date', parse_dates=['Date'])
    futures_df.drop(columns=[futures_df.columns[0], 'BarCount'], inplace=True)
    futures_df.columns = futures_df.columns.str.lower()

    # add tick cols matching on index
    futures_df['tick_high'] = tick_df['high']
    futures_df['tick_low'] = tick_df['low']

    # print the resulting dataframe
    console.print(futures_df)
    console.print(tick_df)


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
        r['open'] = df.Open[s]
        r['high'] = df.High[s:e].max()
        r['low'] = df.Low[s:e].min()
        r['close'] = df.Close[e]
        r['volume'] = df.Volume[s:e].sum()
        vwap = np.average( df.WAP[s:e], weights=df.Volume[s:e] )
        r['vwap'] = round(vwap, 2)
        rows.append(r)
    daily = pd.DataFrame(rows, index=dts)
    daily['change'] = daily['close'].sub(daily['close'].shift())
    daily['day_chg'] = daily['close'].sub(daily['open'])
    daily['range'] = daily['high'].sub(daily['low'])
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


# return boolean index
def find_price_intersection(df, n, start, end):
    return (df['low'].iloc[start:end] <= n) & (df['high'].iloc[start:end] >= n)

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


def main():
    print(f'Hello world {datetime.now()}')
    #df = load_files('/media/niroo/ULTRA/esh1*')
    df = load_files(make_filename('esm1*.csv'))
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

def print_summary(df):
    di = day_index(df)
    dr = rth_index(di)
    console.print('\n--- Day index ---', style='yellow')
    console.print(di)

    console.print('\n--- Day index ---', style='yellow')
    console.print(dr)
    di2 = day_index2(df)
    console.print('\n--- Day index ---', style='yellow')
    console.print(di2)
    
    console.print('\n--- Daily bars ---', style='yellow')
    df2 = aggregate_daily_bars(df, di)
    console.print(df2, style='cyan')
    console.print(f'range min,median,max = {df2['range'].min():.2f} {df2['range'].median():.2f} {df2['range'].max():.2f}', style='green')

    console.print('\n--- RTH bars ---', style='yellow')
    df2 = aggregate_daily_bars(df, dr)
    console.print(df2, style='cyan')
    console.print(f'range min,median,max = {df2['range'].min():.2f} {df2['range'].median():.2f} {df2['range'].max():.2f}', style='green')

def whole_day_concat(fspec, fnout):
    '''combines all files in fspec into one file. takes whole days only'''
    dfs = {f:load_file(f) for f in gb.glob(make_filename(fspec))}
    hw = pd.Timestamp('2020-01-01')
    comb = None
    for fn,df in dfs.items():
        di = day_index(df)
        for i in range(di.shape[0]):
            start = di.iloc[i,0]
            end = di.iloc[i,1]
            if start > hw:
                day = df[(df.index >= start) & (df.index <= end)]
                rows = day.shape[0]
                # 1380 only correct when no daylight savings change
                # for stocks bars=390 and some days might be partial due to half-day holidays
                # if rows == 390:
                if rows == 1380:
                    print(f'{fn} {start} {end} {rows} {hw}')
                    hw = end
                    comb = day if comb is None else pd.concat([comb, day], axis=0, join='outer')
                else:
                    print(f'--skipping incomplete {fn} {start} {end} {rows} ')    
            else:
                print(f'--skipping overlap {fn} {start} {end} {hw}')

    if comb is not None:
        save_df(comb, fnout)
        print_summary(comb)


def test_load(fn):
    dfs = [load_file(e) for e in gb.glob(make_filename(fn))]
    for df in dfs:
        print_summary(df)


def simple_concat(fspec, fnout):
    '''concatenates files into one dataframe skipping duplicate index entries. this only works when minute bars are complete because it picks the first occurence of a time'''
    dfs = [load_file(e) for e in gb.glob(make_filename(fspec))]
    comb = dfs[0]
    for df in dfs[1:]:
        comb = pd.concat([comb, df.loc[~df.index.isin(comb.index)]], axis=0, join='outer')
#    save_df(comb, fnout)
    if not comb.index.is_monotonic_increasing:
        raise ValueError(f'index not monotonic increasing')
    return comb


if __name__ == '__main__':
#    whole_day_concat('esm4*.csv', 'zESM4')
    test_tick()
    # df_es = simple_concat('zesu4*.csv', 'zESU4')
    # print_summary(df_es)
    # df_tick = simple_concat('ztick-nyse*.csv', 'x')
    # di = day_index(df_tick)
    # breakpoint()
#   test_load('zesu4*.csv')