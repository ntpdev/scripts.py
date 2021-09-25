#!/usr/bin/python3
from datetime import datetime, date, time, timedelta
import argparse
import numpy as np
import pandas as pd
import glob as gb
import platform
import sys

class Blotter:

    def __init__(self):
        self.openPositions = []
        self.seqNo = 0
        self.seqDict = {}
        self.trades = []

    def process_trade_log(self, df, skipRows):
        for i,r in df.iterrows():
            if i < skipRows:
                continue
            for x in range(r.Quantity):
                self.process_single_trade(r)
        return pd.DataFrame(self.trades)    

    def process_single_trade(self, r):
        sym = r['Local symbol']
        found = self.find_matching(sym, r['Action'])
        if found == -1:
            if sym not in self.seqDict:
                self.seqDict[sym] = self.seqNo
                self.seqNo += 1
            self.openPositions.append(r)
        else:
            op = self.openPositions.pop(found)
            self.record_trade(op, r)
            if self.count_matching(sym) == 0:
                del self.seqDict[sym]

    def record_trade(self, op, cl):
        trade = {}
        trade['OpTm'] = op.Timestamp
#        trade['ClTm'] = cl.Timestamp
        trade['Seq'] = self.seqDict[op['Local symbol']]
        trade['Symbol'] = op['Local symbol']
        trade['Action'] = op.Action
        trade['Open'] = op.Price
        trade['Close'] = cl.Price
        pts = (cl.Price - op.Price) * (1 if op.Action == 'BOT' else -1)
        prf = self.calc_profit(op.Underlying, pts)
        trade['Points'] = pts
        trade['Profit'] = prf
        trade['Comm'] = 1.04
        trade['Net'] = prf - 1.04
        self.trades.append(trade)
 
    def find_matching(self, symbol, action):
        opening_action = 'SLD' if action == 'BOT' else 'BOT'
        found = -1
        for i,v in enumerate(self.openPositions):
            if v['Local symbol'] == symbol and v['Action'] == opening_action:
                found = i
                break
        return found

    def count_matching(self, symbol):
        return len([1 for x in self.openPositions if x['Local symbol'] == symbol])
    
    def calc_profit(self, symbol, pts):
        if symbol == 'MES':
            return pts * 5
        if symbol == 'MNQ':
            return pts * 2
        return pts


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

def load_file(fname):
    df = pd.read_csv(f'\\Users\\niroo\\OneDrive\\Documents\\{fname}', usecols=[0,1,2,3,4,5,6], parse_dates={'Timestamp' : [5,6]})
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df

parser = argparse.ArgumentParser(description='Process IB trade logs')
parser.add_argument('--skip', metavar='skip', default=0, type=int, help='number of rows to skip')
args = parser.parse_args()

df = pd.concat( [load_file('trades-0913.csv'), load_file('trades-0920.csv')])
#df = load_file('test-trades.csv')
#print(df)
#dfTrades = process_trade_log(df, args.skip)
#print(dfTrades)
#print(dfTrades.groupby(['Seq'])['Ticks'].sum() )
#print_trade_stats(dfTrades)
b = Blotter()
df2 = b.process_trade_log(df, args.skip)
print(df2.tail(19))
print(len(b.openPositions))
