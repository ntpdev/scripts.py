#!/usr/bin/python3
from datetime import date, time, timedelta, datetime
from collections import deque
from dataclasses import dataclass, asdict
from scipy.signal import find_peaks
from pathlib import Path
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
        if acc['volume'] >= minvol or lastbar.loc[i] or i == eur_open or i == rth_open:
            rows.append(acc)
            acc = None
    if acc:
        rows.append(acc)
    df2 = pd.DataFrame(rows)
    df2.set_index('date', inplace=True)
    return df2

def single(dt_fst, fst, period):
    r = {}
    r['date'] = dt_fst
    r['dateCl'] = dt_fst + timedelta(minutes=period)
    r['open'] = fst['open']
    r['high'] = fst['high']
    r['low'] = fst['low']
    r['close'] = fst['close']
    r['volume'] = fst['volume']
    r['vwap'] = fst['vwap']
    if 'ema' in fst:
        r['ema'] = fst['ema']
    return r

def combine(acc, dt_snd, snd, period):
    r = {}
    r['date'] = acc['date']
    r['dateCl'] = dt_snd + timedelta(minutes=period)
    r['open'] = acc['open']
    r['high'] = max(acc['high'], snd['high'])
    r['low'] = min(acc['low'], snd['low'])
    r['close'] = snd['close']
    r['volume'] = acc['volume'] + snd['volume']
    r['vwap'] = snd['vwap']
    if 'ema' in snd:
        r['ema'] = snd['ema']
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


def day_index2(df: pd.DataFrame) -> pd.DataFrame:
    '''create a df [date, first, last, rth_first, rth_last] for each trading day based on gaps in index'''
    first_bar_selector = df.index.diff() != timedelta(minutes=1)
    last_bar_selector = np.roll(first_bar_selector, -1)
    # contiguous time ranges
    idx = pd.DataFrame({'first': df.index[first_bar_selector], 'last': df.index[last_bar_selector]})
    # calculate rth range as offset from open
    rth_start = idx['first'] + timedelta(hours=15, minutes=30)
    rth_end = idx['first'] + timedelta(hours=21, minutes=59)
    nat = np.datetime64("NaT")
    s = np.where(idx['last'] > rth_start, rth_start, nat)
    e = np.where(np.isnat(s), nat, np.minimum(rth_end, idx['last']))
    idx['rth_first'] = s
    idx['rth_last'] = e
    idx['duration'] = ((idx['last'] - idx['first']).dt.total_seconds()) / 60 + 1
    idx.set_index(idx['last'].dt.date, inplace=True)
    idx.index.name = 'date'
    return idx


# def day_index(df):
#     '''create a df [date, openTime, closeTime] for each trading day based on gaps in index'''
#     selector = (df.index.to_series().diff() != timedelta(minutes=1))
#     idx = df[selector].index.array
#     idx_close = df[selector.shift(periods=-1, fill_value=True)].index.array
#     return pd.DataFrame({'openTime':idx, 'closeTime':idx_close}, index=idx_close.date)


# create a df [date, openTime, closeTime] for each rth day from an index of trading days
# def rth_index(day_index):
#     fulldays = day_index[day_index.closeTime - day_index.openTime == timedelta(hours=22, minutes=59)]
#     rth_open = fulldays.openTime + timedelta(hours=15, minutes=30)
#     rth_close = fulldays.openTime + timedelta(hours=21, minutes=59)
#     return pd.DataFrame( {'openTime':rth_open, 'closeTime':rth_close }, index=fulldays.index )


# create a new DF which aggregates bars using a daily index
def aggregate_daily_bars(df, daily_index, start_col, end_col):
    rows = []
    for i,r in daily_index.dropna(subset=[start_col, end_col]).iterrows():
        rows.append(aggregate(df, i, r[start_col], r[end_col]))

    daily = pd.DataFrame(rows)
    daily.set_index('date', inplace=True)
    daily['change'] = daily.close.diff()
    daily['gap'] = daily.open - daily.close.shift()
    daily['day_chg'] = daily.close - daily.open
    daily['range'] = daily.high - daily.low
    daily['strat'] = calc_strat(daily)
    return daily[daily.volume > 10000]


# return a row which aggregates bars between inclusive indexes
def aggregate(df, dt, s, e):
    r = {}
    r['date'] = dt
    r['open'] = df.at[s, 'open']
    r['high'] = df.high[s:e].max()
    r['low'] = df.low[s:e].min()
    r['close'] = df.at[e, 'close']
    r['volume'] = df.volume[s:e].sum()
    # contract expiry is opening price of day so day has no volume
    vwap = 0 if r['volume'] <= 0 else np.average( df.wap[s:e], weights=df.volume[s:e] )
    r['vwap'] = round(vwap, 2)
    return r


def calc_vwap(df):
    is_first_bar = df.index.to_series().diff() != timedelta(minutes=1)
    xs = []
    start = 0
    for i,r in df.iterrows():
        if is_first_bar.loc[i]:
            start = i
        v = np.average( df['wap'].loc[start:i], weights=df['volume'].loc[start:i] )
        xs.append(round(v, 2))
    return pd.Series(xs, df.index)

def calc_strat(df):
    """return a series categorising bar by its strat bar type 0 - inside, 1 up, 2 down, 3 outside"""
    hs = df.high.diff().gt(0)
    ls = df.low.diff().lt(0)
    return hs.astype(int) + ls * 2

def calc_atr(df, n):
    rng = df.high.rolling(n).max() - df.low.rolling(n).min()
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


def make_filename(fname: str) -> Path:
    return Path.home() / 'Documents' / 'data' / fname


def load_files_as_dict(spec: str) -> dict[Path, pd.DataFrame]:
    '''return a dict of paths and data_frames'''
    return {x:load_file(x) for x in sorted((Path.home() / 'Documents' / 'data').glob(spec, case_sensitive=False))}


def load_files(spec: str) -> pd.DataFrame:
    '''load all files matching specified pattern and return a single dataframe'''
    df = pd.concat(load_files_as_dict(spec).values())
    if not df.index.is_monotonic_increasing:
        raise ValueError(f'index not monotonic increasing')
    return df


def load_overlapping_files(spec: str) -> pd.DataFrame:
    '''concatenates files into one dataframe skipping duplicate index entries. this only works when minute bars are complete because it picks the first occurence of a time'''
    comb = None
    for df in load_files_as_dict(spec).values():
        comb = df if comb is None else pd.concat([comb, df.loc[~df.index.isin(comb.index)]], axis=0, join='outer')
    if not comb.index.is_monotonic_increasing:
        raise ValueError(f'index not monotonic increasing')
    return comb


def load_file(fname):
    df = pd.read_csv(fname, parse_dates=['Date'], index_col='Date')
    df.columns = df.columns.str.lower()
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df


def save_df(df, symbol):
    '''save dataframe to csv in original format'''
    idx = day_index(df)
    print(idx)
    fout = make_filename(f'{symbol} {idx.index[0].strftime("%Y%m%d")}.csv')
    print(f'saving {fout}')
    # reverse the parsing operations so the saved csv is the same format
    # drop the first col which is 0,1,2 then make the index date time a standard col
    df2 = df.drop(df.columns[0], axis=1)
    df2.reset_index(names='Date', inplace=True)
    df2['Date'] = df2['Date'].dt.strftime("%Y%m%d %H:%M:%S")
    df2.to_csv(fout)


def create_volume_profile(df, prominence = 40, smoothing_period = 1):
    '''return df of price,volume, peak flag'''
    mx = df['high'].max()
    mn = df['low'].min()
    num_bins = lambda hi,lo : int((hi-lo) * 4 + 1)
    to_bin = lambda p : int((p-mn) * 4)
    to_price = lambda b : b / 4 + mn

    bins = num_bins(mx, mn)
#    print(df.index[0], df.index[-1], mn, mx, bins)

    xs = np.zeros(bins)
    for i,r in df.iterrows():
        x = r['low']
        y = r['high']
        xs[to_bin(x):to_bin(y)+1] += r['volume']/num_bins(y, x)

    if smoothing_period > 1:
        # sma smoothing
        sma_kernel = np.full(smoothing_period, 1 / smoothing_period)
        ys = np.convolve(xs, sma_kernel, 'same')
    else:
        ys = xs
    
    peak_indicies, peak_info = find_peaks(ys, prominence=np.percentile(ys, prominence))
    # pprint(peak_indicies)
    # pprint(to_price(peak_indicies))
    # pprint(peak_info)
    mxs = np.zeros(bins, dtype=bool)
    mxs[peak_indicies] = True
    
    return pd.DataFrame({'volume': xs, 'is_peak':mxs}, index=to_price(np.arange(bins)))
 

@dataclass
class Block:
    dt: datetime # close dt of bar
    open: float
    close: float

def blocks_toDF(blks):
    df = pd.DataFrame(map(asdict, blks))
    df.set_index('dt', inplace=True)
    return df

def calc_tlb(xs, n):
    '''takes a series and a number of blocks for a reversal and returns a DF and the reversal price.'''
    if len(xs) < 2:
        return []
    blks = [Block(xs.index[1], xs.iloc[0], xs.iloc[1])]
    # queue of last n+1 closes
    q = deque(xs[0:2], maxlen=n+1)
    dirn = 1 if xs.iloc[1] > xs.iloc[0] else -1
    for dt,x in xs[2:].items():
        last_cl = q[-1]
        rev = q[0]
        if (dirn == 1 and x > last_cl) or (dirn == -1 and x < last_cl):
            blks.append(Block(dt, last_cl, x))
            q.append(x)
        elif (dirn == 1 and x < rev) or (dirn == -1 and x > rev):
            #print(f'rev {x} {q}')
            prev_cl = q[-2]
            blks.append(Block(dt, prev_cl , x))
            q.clear()
            q.append(prev_cl)
            q.append(x)
            dirn *= -1
    print(f'{"uptrend" if dirn == 1 else "downtrend"} reversal price {q[0]}')

    return blocks_toDF(blks), q[0]


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