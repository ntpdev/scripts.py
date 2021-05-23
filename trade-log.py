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

def processSingleTrade(openPositions, trades, r):
    if len(openPositions) == 0 or openPositions[0]['Action'] == r.Action:
        trade = {}
        trade['Action'] = r.Action
        trade['Open'] = r.Price
        dt = parseTimeDate(r.Time, r.Date)
        trade['Date'] = dt.date()
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
        trade['Ticks'] = 4 * trade['Points']
        trade['Profit'] = 1.25 * trade['Ticks']
        trades.append(trade)

def print_trade_stats(dfTrades):
    wins = dfTrades.Ticks[dfTrades.Ticks > 0].count()
    sumWins = dfTrades.Ticks[dfTrades.Ticks > 0].sum()
    loses = dfTrades.Ticks[dfTrades.Ticks < 0].count()
    sumLoses = dfTrades.Ticks[dfTrades.Ticks < 0].sum()
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
    trades = []

    for i,r in df.iterrows():
        if i < skipRows:
            continue
        for x in range(r.Quantity):
            processSingleTrade(openPositions, trades, r)
    dfTrades = pd.DataFrame(trades)
    if len(openPositions) > 0:
        print(f'** Unmatched {len(openPositions)}')

    return dfTrades[['Date','Action','Open','Close','Points','Ticks','Profit','OpTm','ClTm','Duration']]

parser = argparse.ArgumentParser(description='Process IB trade logs')
parser.add_argument('--skip', metavar='skip', default=2, type=int, help='number of rows to skip')
args = parser.parse_args()

df = pd.read_csv('\\Users\\niroo\\OneDrive\\Documents\\trades0517.csv')
print(df)
dfTrades = process_trade_log(df, args.skip)
print(dfTrades)
print_trade_stats(dfTrades)