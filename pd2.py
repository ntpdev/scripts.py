#!/usr/bin/python3
from datetime import datetime
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
#xs = glob.glob('/media/niroo/ULTRA/esh1*')
xs = glob.glob('D:\esh1*.csv')
for s in xs:
    print(s)
    df = pd.read_csv(s, index_col='Date')
    print(df)
    calc_hilo(df)

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
fig.show()