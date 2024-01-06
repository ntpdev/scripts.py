#!/usr/bin/python3
from datetime import date, time, timedelta, datetime
from collections import deque
import numpy as np
import pandas as pd
import glob as gb
import platform

# import tsutils as ts
# df = ts.load_file(ts.make_filename('esu1 20210705.csv'))

# Time series utility functions

def find_initial_swing(s, perc_rev):
    '''return direction and start index and end index of the first incomplete swing.'''
    hw = s.iloc[0]
    hwi = 0
    lw = s.iloc[0]
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
                return (1, lwi, hwi)
            else:
                return (-1, hwi, lwi)
    return (0, 0, 0)

def find_swings(s, perc_rev):
    '''return df of swings. The final row is the current extreme of the in-progress swing'''
    dirn, start_index, end_index = find_initial_swing(s, perc_rev)
    xs = []
    if dirn == 0:
        return xs
    xs.append(start_index)
    extm = s.iloc[end_index]
    extm_index = end_index
    for i in range(end_index+1, s.size):
        x = s.iat[i]
        if dirn == 1:
            if x > extm:
                extm = x
                extm_index = i
#                print(f'new hi {extm_index} {extm}')
            elif pdiff(extm, x, perc_rev):
#                    print(f'reversal {x}')
                    xs.append(extm_index)
                    dirn = -1
                    extm = x
                    extm_index = i
        else:
            if x < extm:
                extm = x
                extm_index = i
#                print(f'new lo {extm_index} {extm}')
            elif pdiff(extm, x, perc_rev):
#                    print(f'reversal {x}')
                    xs.append(extm_index)
                    dirn = 1
                    extm = x
                    extm_index = i
    xs.append(extm_index) # store the last unconfirmed extreme
    day_index = np.array(xs)
    swing = s.iloc[day_index[:-1]]
    ends = s.iloc[day_index[1:]]
    df = pd.DataFrame({'start': swing, 'end': ends.to_numpy(), 'dtend': ends.index, 'days': np.diff(day_index) + 1})
    df['change'] = ((df['end'] / df['start'] -1.) * 100.).round(2)
    df['mae'] = [round(calculate_mae(s[x:y+1]), 2) for x,y in zip(xs, xs[1:])]
    return df


def calculate_mae(s):
    '''calculate the mae for the series'''
    if s.iloc[0] < s.iloc[-1]: # uptrend
        mx = s.expanding().max()
        excursion = (s - mx) / mx
        return excursion.min() * 100.
    mx = s.expanding().min()
    excursion = (s - mx) / mx
    return excursion.max() * 100.


# return true if perc diff gt
def pdiff(s, e, p):
    return 100 * abs(e / s - 1) >= p

# inclusive end
def aggregate(df):
    acc = {}
    for i,r in df.iterrows():
        acc = combine(acc, i, r, 1) if acc else single(i,r,1) 
    return acc


def aggregateMinVolume(df, minvol):
    rows = []
    acc = {}
#    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    selector = df.index.to_series().diff() != timedelta(minutes=1)
    openbar = (df.index.minute == 0) & selector
    lastbar = selector.shift(-1, fill_value=True)
    eur_open = date(2021,1,1)
    rth_open = date(2021,1,1)
    for i,r in df.iterrows():
        if openbar.loc[i]:
            eur_open = i + timedelta(hours=8, minutes=59)
            rth_open = i + timedelta(hours=15, minutes=29)
        acc = combine(acc, i, r, 1) if acc else single(i,r,1)
        if acc['Volume'] >= minvol or lastbar.loc[i] or i == eur_open or i == rth_open:
            rows.append(acc)
            acc = None
    if acc:
        rows.append(acc)
    df2 = pd.DataFrame(rows)
    df2.set_index('Date', inplace=True)
    return df2

def single(dt_fst, fst, period):
    r = {}
    r['Date'] = dt_fst
    r['DateCl'] = dt_fst + timedelta(minutes=period)
    r['Open'] = fst['Open']
    r['High'] = fst['High']
    r['Low'] = fst['Low']
    r['Close'] = fst['Close']
    r['Volume'] = fst['Volume']
    r['VWAP'] = fst['VWAP']
    if 'EMA' in fst:
        r['EMA'] = fst['EMA']
    return r

def combine(acc, dt_snd, snd, period):
    r = {}
    r['Date'] = acc['Date']
    r['DateCl'] = dt_snd + timedelta(minutes=period)
    r['Open'] = acc['Open']
    r['High'] = max(acc['High'], snd['High'])
    r['Low'] = min(acc['Low'], snd['Low'])
    r['Close'] = snd['Close']
    r['Volume'] = acc['Volume'] + snd['Volume']
    r['VWAP'] = snd['VWAP']
    if 'EMA' in snd:
        r['EMA'] = snd['EMA']
    return r

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

# create a df [date, openTime, closeTime] for each trading day
def day_index(df):
    selector = (df.index.minute == 0) & (df.index.to_series().diff() != timedelta(minutes=1))
    idx = df[selector].index.array
    idx_close = df[selector.shift(periods=-1, fill_value=True)].index.array
    return pd.DataFrame({'openTime':idx, 'closeTime':idx_close}, index=idx_close.date)


# create a df [date, openTime, closeTime] for each rth day from an index of trading days
def rth_index(day_index):
    fulldays = day_index[day_index.closeTime - day_index.openTime == timedelta(hours=22, minutes=59)]
    rth_open = fulldays.openTime + timedelta(hours=15, minutes=30)
    rth_close = fulldays.openTime + timedelta(hours=21, minutes=59)
    return pd.DataFrame( {'openTime':rth_open, 'closeTime':rth_close }, index=fulldays.index )


# create a new DF which aggregates bars using a daily index
def aggregate_daily_bars(df, daily_index):
    rows = []
    for i,r in daily_index.iterrows():
        rows.append(aggregate(df, r['openTime'], r['closeTime']))

    daily = pd.DataFrame(rows, index=daily_index.index)
    daily['Change'] = daily.Close.diff()
    daily['Gap'] = daily.Open - daily.Close.shift()
    daily['DayChg'] = daily.Close - daily.Open
    daily['Range'] = daily.High - daily.Low
    return daily[daily.Volume > 500000]


# return a row which aggregates bars between inclusive indexes
def aggregate(df, s, e):
    r = {}
    r['Open'] = df.at[s, 'Open']
    r['High'] = df.High[s:e].max()
    r['Low'] = df.Low[s:e].min()
    r['Close'] = df.at[e, 'Close']
    r['Volume'] = df.Volume[s:e].sum()
    # contract expiry is opening price of day so day has no volume
    vwap = 0 if r['Volume'] <= 0 else np.average( df.WAP[s:e], weights=df.Volume[s:e] )
    r['VWAP'] = round(vwap, 2)
    return r


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

def calc_strat(df):
    """return a series categorising bar by its strat bar type 0 - inside, 1 up, 2 down, 3 outside"""
    hs = df.High.diff().gt(0)
    ls = df.Low.diff().lt(0)
    return hs.astype(int) + ls * 2

def calc_atr(df, n):
    rng = df.High.rolling(n).max() - df.Low.rolling(n).min()
    df2 = pd.DataFrame( {'tm':df.index.time, 'rng':rng}, index=rng.index )
    return df2.groupby('tm').rng.agg('mean')

def make_threeLB(x, xs):
    if x > xs[0]:
        xs.append(x)
    if len(xs) > 2 and x < xs[2]:
        xs = deque()
        xs.append(x)
    return xs

def to_date(timestmp):
    return timestmp.to_pydatetime().date()

def make_filename(fname):
    p = '/media/niroo/ULTRA/' if platform.system() == 'Linux' else 'c:\\temp\\ultra\\'
    return p + fname

def load_files(fname):
    dfs = [load_file(e) for e in gb.glob(fname)]
    return pd.concat(dfs)

def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['Date'], index_col='Date')
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df

class LineBreak:

    def __init__(self, n):
        self.reversalBocksLength = n
        self.dirn = 0
        self.lines = deque(maxlen = n + 1)
        self.blocks = []

    def append(self, x, dt):
        if len(self.lines) < self.reversalBocksLength:
            self._appendBlock(x, dt)
        else:
            high = max(self.lines)
            low = min(self.lines)
            if x > high or x < low:
                # if reversal add prior close to queue of lines
                if (x > high and self.dirn == -1) or (x < low and self.dirn == 1):
#                    print(f'reversal adding {self.lines[-2]}')
                    self.lines.append(self.lines[-2])
                self._appendBlock(x, dt)
    
    def asDataFrame(self):
        return pd.DataFrame(self.blocks)

# add closing price to lines queue and if there is at least 1 prior line add a block
# update direction of the last block in self.dirn
    def _appendBlock(self, x, dt):
        if len(self.lines) > 0:
            last = self.lines[-1]          
            self.dirn = 1 if x > last else -1                
            self.lines.append(x)
            block = {}
            block['date'] = dt
            block['open'] = last
            block['close'] = x
            block['dirn'] = self.dirn
            self.blocks.append(block)
        else:
            self.lines.append(x)

# first block is wrong
    def test(self):
        cls = [135, 132, 128, 133, 130, 130, 132, 134, 139, 137, 145, 158, 147, 143, 150, 149, 160, 164, 167, 156, 165, 168,
        171,173,169,177,180,176,170,175,179,173,170,170,168,165,171,175,179,175]
        for c in cls:
            self.append(c)
            print(f'{c} {self.lines}')
        df = self.asDataFrame()
        print(df)

# to reload a module then reload the name/alias of the imported module
# from importlib import reload
# reload(ts)