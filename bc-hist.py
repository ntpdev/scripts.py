#!/usr/bin/python3
import argparse
import re
from datetime import date, time, timedelta, datetime
from collections import deque
import numpy as np
import pandas as pd
from glob import glob
import tsutils as ts


# calculate percentage change from initial value scaled to 100%
def calc_pct(ser):
    first = ser.iat[0]
    return ((ser / first - 1.0) * 100).round(2)

def find_initial_swing(s, perc_rev):
    hw = s[0]
    hwi = 0
    lw = s[0]
    lwi = 0
    for i in range(s.size):
        x = s.iat[i]
        if x > hw:
            hw = x
            hwi = i
        elif x < lw:
            lw = x
            lwi = i
        if pdiff(lw, hw, perc_rev):
            if lwi < hwi:
                return (1, lwi)
            else:
                return (-1, hwi)
    return (0, 0)

def find_swings(s, perc_rev):
    dirn, i = find_initial_swing(s, perc_rev)
    xs = []
    if dirn == 0:
        return xs
    xs.append(i)
    hw = s[i]
    hwi = i
    lw = s[i]
    lwi = i
    for i in range(1, s.size):
        x = s.iat[i]
#        print(f'{x} loww {lw} hiw {hw} dirn {dirn}')
        if dirn == 1:
            if x > hw:
                hw = x
                lw = x
                hwi = i
                lwi = i
            elif x < lw:
                lw = x
                if pdiff(hw, lw, perc_rev):
                    xs.append(hw)
                    hw = x
                    hwi = i
                    dirn = -1
        else:
            if x < lw:
                lw = x
                hw = x
                hwi = i
                lwi = i
            elif x > hw:
                hw = x
                if pdiff(lw, hw, perc_rev):
                    xs.append(lw)
                    lw = x
                    lwi = i
                    dirn = 1
#        print(f'-- loww {lw} lwi {lwi} hiw {hw} hwi {hwi} dirn {dirn}')
    xs.append(s.iat[s.size - 1])
    ys = np.array(xs)
    print(ys)
    ds = (ys[1:] / ys[:-1] - 1) * 100
    print(ds)
    return xs

def find_swings2(s, perc_rev):
    i = 0
    hw = s[i]
    hwi = i
    for i in range(1, s.size):
        x = s.iat[i]


# return true if perc diff gt
def pdiff(s, e, p):
    return 100 * abs(e / s - 1) >= p

def print_range(df, n):
    rng = df.High - df.Low
    h = df.High[-n:].max()
    l = df.Low[-n:].min()
    m = df.Last[-n:].mean()
    c = df.Last.iat[-1]
    r = 100*(c-l)/(h-l)
    # atr not calculated with prior low
    atr = rng[-n:].mean()
    sd = df.PctChg[-n:].std()
    row = {'n' : n, 'high' : h, 'low' : l, 'mean' : m, 'pctrng' : r, 'stdev' : sd}
    return row

def print_stats(df):
    xs = []
    xs.append(print_range(df, 20))
    xs.append(print_range(df, 50))
    xs.append(print_range(df, df.shape[0]))
    df_stats = pd.DataFrame(xs)
    df_stats.set_index('n', inplace=True)
    print(df_stats)

    cl = df.Last.iat[-1]
    mx = df.Last.max()
    mn = df.Last.min()
    pmx = (cl / mx - 1) * 100
    pmn = (cl / mn - 1) * 100
    print(f'close {cl} up from low {mn} {pmn:.2f}%, down from high {mx} {pmx:.2f}%')

def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['Time'], index_col='Time', skipfooter=1, engine='python')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
#    return df[::-1].reset_index(drop=False)
    return df[::-1]

def export_file(df, outfile):
    fn = ts.make_filename(outfile)
    print(f'saving file {outfile} {len(df)}')
    df.to_csv(outfile)

# spy_price-history-12-30-2022.csv
def parse_date(fn):
    m = re.search('\d\d-\d\d-\d\d\d\d', fn)
    return datetime.strptime(m.group(0), '%m-%d-%Y').date()

def find_file(symbol):
    files = glob(f'/users/niroo/downloads/{symbol}_price*.csv')
    files.sort(key = lambda e: parse_date(e))
    print(files)
    return files[-1]

parser = argparse.ArgumentParser(description='Barchart history')
parser.add_argument('fname', help='Input file')
args = parser.parse_args()

df = load_file( find_file(args.fname) if (len(args.fname) < 5) else args.fname )
df['HiLo'] = ts.calc_hilo(df['Last'])
df['PctChg'] = df['Last'].pct_change().round(4) * 100
df['LastPct'] = calc_pct(df['Last'])
find_swings(df['Last'], 5.0)
export_file(df, args.fname.split('_')[0] + '.csv')
print(df)
print_stats(df)

lb = ts.LineBreak(3)
for i,r in df.iterrows():
    lb.append(r['Last'], i)
df3 = lb.asDataFrame()
print(df3)
export_file(df3, '3lb.csv')
