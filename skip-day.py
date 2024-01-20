#!/usr/bin/python3
import argparse
from datetime import date, time, timedelta, datetime
import numpy as np
import pandas as pd
import glob as gb
import platform
from pathlib import Path
import tsutils as ts

def make_output_name(p, dt):
    sym = p.name.split(' ')[0]
    d = dt.strftime('%Y%m%d')
    s = f'{sym} {d}.csv'
    return p.parent / s

parser = argparse.ArgumentParser(description='Remove day from price history')
parser.add_argument('fname', help='Input file')
args = parser.parse_args()

p = Path(args.fname)
if p.exists():
    fname = str(p)
    df = pd.read_csv(fname, parse_dates=['Date'], index_col=0)
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    outfile = make_output_name(p, date(2022,4,11))
    df2 = df[1380:]
    print(f'saving {outfile} {len(df2)}')
    df2.to_csv(str(outfile))
