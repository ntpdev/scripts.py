#!/usr/bin/python3
from datetime import datetime, date, time, timedelta
from collections import deque
import argparse
import numpy as np
import pandas as pd
import glob as gb
import platform
import sys

def parseTimeDate(sTime, sDate):
     return datetime.strptime( f'{sDate} {sTime}', '%Y%m%d  %H:%M:%S')

def processSingleTrade(openPositions, trades, seqNo, r):
    if len(openPositions) == 0 or openPositions[0]['Action'] == r.Action:
        trade = {}
        trade['Action'] = r.Action
        trade['Open'] = r.Price
        dt = parseTimeDate(r.Time, r.Date)
        trade['Date'] = dt.date()
        trade['Seq'] = seqNo
        trade['Symbol'] = r['Local symbol']
        trade['OpTm'] = dt
        openPositions.append(trade)
    else:
        trade = openPositions.popleft()
        open = trade['Open']
        trade['Close'] = r.Price
        trade['ClTm'] = parseTimeDate(r.Time, r.Date)
        trade['Duration'] = round((trade['ClTm'] - trade['OpTm']).total_seconds() / 60.0,2)
        if trade['Action'] == 'BOT':
            trade['Points'] = trade['Close'] - trade['Open']
        else:
            trade['Points'] = trade['Open'] - trade['Close']
        if trade['Symbol'] != r['Local symbol']:
            print(f'Symbol mismatch for row {r}')
            sys.exit(0)
        trade['Ticks'] = 4 * trade['Points']
        trade['Profit'] = 1.25 * trade['Ticks']
        trades.append(trade)
    return seqNo if len(openPositions) != 0 else seqNo + 1

def print_trade_stats(dfTrades):
    wins = dfTrades.Ticks[dfTrades.Ticks > 3].count()
    sumWins = dfTrades.Ticks[dfTrades.Ticks > 3].sum()
    loses = dfTrades.Ticks[dfTrades.Ticks < -3].count()
    sumLoses = dfTrades.Ticks[dfTrades.Ticks < -3].sum()
    durWin = dfTrades.Duration[dfTrades.Ticks > 0].sum()
    durLoss = dfTrades.Duration[dfTrades.Ticks < 0].sum()
    winPerc = 100 * wins / (wins + loses)
    avgWin = sumWins / wins
    avgLoss = sumLoses / loses
    ratio = avgWin / -avgLoss
    avgWinDuration = durWin / wins
    avgLossDuration = durLoss / loses
    cost = 1.04 * len(dfTrades)

    print(f'Contracts: {len(dfTrades)} Ticks: {dfTrades.Ticks.sum()} P/L: ${dfTrades.Profit.sum():.2f} Commisions: ${cost:.2f}')
    print(f'wins: {wins} loses: {loses} win%: {winPerc:.0f}%')
    print(f'wins: {sumWins} loses: {sumLoses}')
    print(f'avg w: {avgWin:.1f} avg l: {avgLoss:.1f} ratio: {ratio:.1f}')
    print(f'avg w time: {avgWinDuration:.1f} avg l time: {avgLossDuration:.1f}')

def process_trade_log(df, skipRows):
    openPositions = deque()
    seqNo = 0
    trades = []

    for i,r in df.iterrows():
        if i < skipRows:
            continue
        for x in range(r.Quantity):
            seqNo = processSingleTrade(openPositions, trades, seqNo, r)
    dfTrades = pd.DataFrame(trades)
    if len(openPositions) > 0:
        print(f'** Unmatched {len(openPositions)}')

#    return dfTrades[['Date','Action', 'Sequence','Symbol', 'Open','Close','Points','Ticks','Profit','OpTm','ClTm','Duration']]
    return dfTrades[['Date','Action', 'Seq','Symbol', 'Open','Close','Points','Ticks','Profit','Duration']]

parser = argparse.ArgumentParser(description='Process IB trade logs')
parser.add_argument('--skip', metavar='skip', default=0, type=int, help='number of rows to skip')
args = parser.parse_args()

df = pd.read_csv('\\Users\\niroo\\OneDrive\\Documents\\trades.20210531.csv')
#df = pd.read_csv('\\Users\\niroo\\OneDrive\\Documents\\trades2.csv')
print(df)
dfTrades = process_trade_log(df, args.skip)
print(dfTrades)
print(dfTrades.groupby(['Seq'])['Ticks'].sum() )
print_trade_stats(dfTrades)