#!/usr/bin/python3
import argparse
from datetime import date, time, timedelta, datetime
from collections import deque
import numpy as np
import pandas as pd
import glob as gb
import platform
import tsutils as ts

def count_back(xs, i):
    current = xs.iloc[i]
    c = 0
    for k in range(i-1, -1, -1):
        prev = xs.iloc[k]
        if c > 0:
            if current >= prev:
                c += 1
            else:
                break
        elif c < 0:
            if current <= prev:
                c -= 1
            else:
                break
        else:
            c = 1 if current >= prev else -1

    return c

def calc_hilo(ser):
    cs = []
    cs.append(0)
    for i in range(1, ser.size):
        cs.append(count_back(ser, i))
    return pd.Series(cs, ser.index)

def print_range(df, n):
    h = df.High[-n:].max()
    l = df.Low[-n:].min()
    c = df.Last.iat[-1]
    r = 100*(c-l)/(h-l)
    print(f'{n}d range {l:.2f} {h:.1f} {r:.1f}%')

def print_stats(df, n):
    print(df[-49:])
    xs = df.PctChg.rolling(n).std()
    print(f'{n}d stddev {xs.iat[-1]:.2f}%')
    print_range(df, 20)
    print_range(df, 50)

def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['Time'], index_col='Time', skipfooter=1, engine='python')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
#    return df[::-1].reset_index(drop=False)
    return df[::-1]

def export_file(df, outfile):
    fn = ts.make_filename(outfile)
    print(f'saving file {outfile} {len(df)}')
    df.to_csv(outfile)

parser = argparse.ArgumentParser(description='Barchart history')
parser.add_argument('fname', help='Input file')
args = parser.parse_args()

df = load_file(args.fname)
df['HiLo'] = calc_hilo(df['Last'])
df['PctChg'] = df['Last'].pct_change().round(4) * 100
export_file(df, args.fname.split('_')[0] + '.csv')
print(df)
print_stats(df, 50)

lb = ts.LineBreak(3)
for i,r in df.iterrows():
    lb.append(r['Last'], i)
df3 = lb.asDataFrame()
print(df3)
export_file(df3, '3lb.csv')
