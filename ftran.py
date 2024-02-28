#!/usr/bin/python3
import numpy as np
import pandas as pd

def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['Time'], index_col='Time', skipfooter=1, engine='python')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
#    return df[::-1].reset_index(drop=False)
    return df[::-1]

def norm(fname):
    df = load_file(fname)
    v = df['Volume']
    vmean = v.rolling(20).mean()
    vstd = v.rolling(20).std()
    vnorm = (v - vmean) / vstd
    first = df['Last'].iat[0]
    pct = (df['Last'] / first - 1.0) * 100
    xs = {'volume' : v , 'mean' : vmean, 'stdev' : vstd, 'vnorm' : vnorm, 'pct' : pct}
    df2 = pd.DataFrame(xs)
    print(df2)
    xs = []

def test1():
    fn = 'f:\\a.txt'

    with open(fn, 'r') as f:
        text = f.readlines()

    a = []
    for s in text:
        a.append(s.strip())

    a.sort()

    with open(fn + '.out', 'w') as fout:
        for s in a:
            fout.write(s + '\n')
        
        x = a[0]
        for s in a[1:]:
            x = x + ',' + s
        fout.write(x + '\n')

def updateCsv():
    fn = 'c:\\temp\\ultra\\esz3'
    df = pd.read_csv(fn, parse_dates=['Date'], index_col='Date', engine='python')
    print(f'loaded {fn} {df.shape[0]} {df.shape[1]}')
    print(df)
    # save with date index as ISO8601 assuming input is UTC
    df.to_csv(fn + 'b', columns=["Open","High","Low","Close","Volume"], date_format="%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    #norm('C:\\Users\\niroo\\Downloads\\spy_price-history-10-15-2022.csv')
    updateCsv()