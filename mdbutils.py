#!/usr/bin/python3

from datetime import datetime, date, time, timedelta
from dataclasses import dataclass
import numpy as np
import pandas as pd
import math
from pymongo.mongo_client import MongoClient
from bisect import bisect_right
from rich.console import Console
from rich.pretty import pprint
import tsutils as ts

console = Console()


def load_summary(collection):
    '''return a summary of all symbols in the collection. df [symbol, count, start, end]. symbol is index'''
    cursor = collection.aggregate([
          {"$group": {"_id": "$symbol",
                      "count": {"$sum": 1},
          			  "start": {"$first": "$timestamp"},
                      "end": {"$last": "$timestamp"}
                     }},                 
          {"$sort" : {"_id": 1}}
    ])
    xs = []
    for r in cursor:
        xs.append(
            { "symbol" : r['_id'],
              "count" : r['count'],
              "start": r['start'],
              "end": r['end'] } )
    df = pd.DataFrame(xs)
    df.set_index('symbol', inplace=True)
    return df


def load_timeseries(collection, symbol, tmStart, tmEnd):
    # tz difference changes normal 23:00 but can be 22:00 so load from 10pm prior day
    print(f'loading {symbol} {tmStart} to {tmEnd}')
    #plot_normvol(dfDays)
    c = collection.find(filter = {'symbol' : symbol, 'timestamp' : {'$gte': tmStart, '$lt': tmEnd}}, sort = ['timestamp'])
    rows = []
    for d in c:
        rows.append( {
                'date': d['timestamp'],
                'open': d['open'],
                'high': d['high'],
                'low':  d['low'],
                'close': d['close'],
                'volume': d['volume'],
                'vwap': d['vwap'] } )
    df = pd.DataFrame(rows)
    df.set_index('date', inplace=True)
    df['ema'] = df.close.ewm(span=90, adjust=False).mean()
# TODO change tsutils to return lower case column names    df['strat'] = ts.calc_strat(df)
    return df

def load_trading_days(collection, symbol, minVol):
    """return df of complete trading days. the bars are aggregated by calendar date [date, bar-count, volume, normalised-volume]"""
    cursor = collection.aggregate(
        [{'$match': {'symbol': symbol}},
         {'$group': {
            '_id'   : {'$dateTrunc': {'date': '$timestamp', 'unit': 'day'}}, 
            'count' : {'$sum': 1},
            'volume': {'$sum': '$volume'} } },
         {'$match': {'volume': {'$gte': minVol}}},
         {'$sort': {'_id': 1}}] )
    df = pd.DataFrame(list(cursor))
    v = df.volume
    df['normv'] = (v - v.mean()) / v.std()
    df.set_index('_id', inplace=True)
    df.index.rename('date', inplace=True)
    return df


def load_gaps(collection, symbol, gap_mins):
    '''return table of gaps in m1 time series. last bar of day and first bar of next day [last_bar, first_bar, gap]'''
    cursor = collection.aggregate([
      {
        '$match': { 'symbol': symbol },
      },
      {
        '$setWindowFields': {
          'sortBy': { 'timestamp': 1 },
          'output': {
            'lastTm': {
              '$shift': {
                'output': "$timestamp",
                'by': -1,
              },
            },
          },
        },
      },
      {
        '$project': {
          'timestamp': 1,
          'lastTm': 1,
          'gap': {
            '$dateDiff': {
              'startDate': "$lastTm",
              'endDate': "$timestamp",
              'unit': "minute",
            },
          },
        },
      },
      {
        '$match': { 'gap': { '$gte': gap_mins } }
      }
    ])
    xs = []
    for r in cursor:
        xs.append({
            'last_bar': r['lastTm'],
            'first_bar': r['timestamp'],
            'gap': r['gap']
        })
    return pd.DataFrame(xs)


def find_date_range(xs, x, n):
    '''given a sorted list xs return start,end values for n elements less than or equal to x. If n is negative x will be the last item.'''
    a,b = find_index_range(xs, x, n)
    return xs[a], xs[b]


def find_index_range(xs, x, n):
    '''given a sorted list xs return start,end index for n elements less than or equal to x. If n is negative x will be the last item.'''
    i = bisect_right(xs, x)
    if i < 1:
        raise ValueError(f'{x} is before first index entry {xs[0]}')
    i -= 1
    end = i + int(math.copysign(abs(n)-1, n))
    if n < 0:
        i, end = end, i
    return max(i, 0), min(end, len(xs)-1)


def find_datetime_range(dfTradeDays, day, n):
    start, end = find_date_range(dfTradeDays.index.tolist(), day, n)
    return dfTradeDays.at[start, 'start'], dfTradeDays.at[end, 'end']


def make_trade_dates(dfSummary, dfGaps, symbol):
    '''build df of [trade_date, start, end, rth_start] where range is [start, end)'''
    e = pd.concat([dfGaps['last_bar'], pd.Series(dfSummary.at[symbol,'end'])], ignore_index=True)
    s = pd.concat([pd.Series(dfSummary.at[symbol, 'start']), dfGaps['first_bar']], ignore_index=True)
    # rth_end = start + 1320 - the day may end earlier but this is fine if using a query
    df = pd.DataFrame({'start': s, 'end': e + pd.Timedelta(minutes=1), 'rth_start': s + pd.Timedelta(minutes=930)})
    df.set_index(e.dt.date, inplace=True)
    df.index.name = 'date'
    return df


def calculate_trading_hours(dfTradeDays, dt, range_name):
    '''return start and end inclusive datetime for a given date and range name. range_name is 'rth' or 'glbx' or 'day'.'''
    try:
        st = dfTradeDays.at[dt, 'start']
        end = dfTradeDays.at[dt, 'end'] # dfTradeDays has exclusive end
        if range_name == 'rth':
            return st + pd.Timedelta(minutes=930), st + pd.Timedelta(minutes=1319)
        elif range_name == 'glbx':
            return st, st + pd.Timedelta(minutes=929)
        return st, end
    except KeyError:
        print(f'KeyError: {dt} not found in dataframe')
    return None


def create_day_summary(df, df_di):
  console.print(df_di)
  xs = []
  for i,r in df_di.iterrows():
    openTime = r['first']
    rthOpen = r['rth_first']
    euClose = min(r['last'], rthOpen - pd.Timedelta(minutes=1))
    glbx_df = df[openTime:euClose]
    rth_hi, rth_lo, rth_hi_tm, rth_lo_tm, rth_open, rth_close, rth_fhi, rth_flo = pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA, pd.NA
    if not pd.isnull(rthOpen):
      rth_df = df[rthOpen:r['rth_last']]
      rth_hi = rth_df['high'].max()
      rth_hi_tm = rth_df['high'].idxmax()
      rth_lo = rth_df['low'].min()
      rth_lo_tm = rth_df['low'].idxmin()
      rth_open = rth_df.iat[0, rth_df.columns.get_loc('open')]
      rth_close = rth_df.iat[-1, rth_df.columns.get_loc('close')]

      rth_h1_last = rthOpen + pd.Timedelta(minutes=59)
      rth_h1_df = df[rthOpen:rth_h1_last]
      rth_fhi = rth_h1_df['high'].max()
      rth_flo = rth_h1_df['low'].min()

    xs.append({'date' : i,
      'glbx_high': glbx_df['high'].max(),
      'glbx_low': glbx_df['low'].min(),
      'rth_open': rth_open,
      'rth_high': rth_hi,
      'rth_low': rth_lo,
      'close': rth_close,
      'rth_high_tm': rth_hi_tm,
      'rth_low_tm': rth_lo_tm,
      'rth_h1_high': rth_fhi,
      'rth_h1_low': rth_flo
    })

  day_summary_df = pd.DataFrame(xs)
  day_summary_df.set_index('date', inplace=True)
  console.print(day_summary_df)
  return day_summary_df


def load_price_history(symbol, dt, n = 1):
    client = MongoClient('localhost', 27017)
    collection = client['futures'].m1
    dfSummary = load_summary(collection)
    # console.print(dfSummary)
    dfGaps = load_gaps(collection, symbol, 30)
    # console.print(dfGaps)
    dfTradeDays = make_trade_dates(dfSummary, dfGaps, symbol)
    # console.print(dfTradeDays)
    s,e = find_datetime_range(dfTradeDays, dt, n-1 if n < 0 else n)
    # console.print(s, e)
    return load_timeseries(collection, symbol, s, e)


def main(symbol: str, dt: date = None):
    if dt is None:
        dt = date.today()
    client = MongoClient('localhost', 27017)
    collection = client['futures'].m1
    dfDays = load_trading_days(collection, symbol, 100000)
    console.print(dfDays)
    console.print(find_index_range(dfDays.index.date.tolist(), date(2024, 7, 8), 1))
    # console.print(find_index_range(dfDays.index.date.tolist(), date(2024, 7, 8), 3))
    # console.print(find_date_range(dfDays.index.date.tolist(), date(2024, 7, 8), -2))
    dfSummary = load_summary(collection)
    console.print(dfSummary)
    dfGaps = load_gaps(collection, symbol, 30)
    console.print(dfGaps)
    dfTradeDays = make_trade_dates(dfSummary, dfGaps, symbol)
    console.print(dfTradeDays)
    tms,tme = find_datetime_range(dfTradeDays, dt, -3)
    df = load_timeseries(collection, symbol, tms, tme)
    tms, tme = calculate_trading_hours(dfTradeDays, tme.date(), 'rth')
    console.print(df[tms:tme].head())
    console.print(df[tms:tme].tail())
    df_di = ts.day_index2(df)
    console.print(df_di)
    console.print(create_day_summary(df, df_di))


if __name__ == '__main__':
#    whole_day_concat('esm4*.csv', 'zESM4')
    main('esu4')