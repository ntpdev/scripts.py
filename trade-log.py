#!/usr/bin/python3
from datetime import datetime, date, time, timedelta
import argparse
import numpy as np
import pandas as pd
import glob as gb
import platform
import sys
from rich.console import Console
from rich.pretty import pprint

console = Console()

class Blotter:

    def __init__(self):
        self.openPositions = []
        self.nextSeqNo = 0
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
        sym = r['Symbol']
        found = self.find_matching(sym, r['Action'])
        if found == -1:
            if sym not in self.seqDict:
                self.seqDict[sym] = self.nextSeqNo
                self.nextSeqNo += 1
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
        trade['Seq'] = self.seqDict[op['Symbol']]
        trade['Symbol'] = op['Symbol']
        trade['Action'] = op.Action
        trade['Open'] = op.Price
        trade['Close'] = cl.Price
        pts = (cl.Price - op.Price) * (1 if op.Action == 'BOT' else -1)
        prf = self.calc_profit(op['Fin Instrument'][:3], pts)
        trade['Points'] = pts
        trade['Profit'] = prf
        trade['Comm'] = 1.24
        trade['Net'] = prf - 1.24
        self.trades.append(trade)
 
    def find_matching(self, symbol, action):
        opening_action = 'SLD' if action == 'BOT' else 'BOT'
        found = -1
        for i,v in enumerate(self.openPositions):
            if v['Symbol'] == symbol and v['Action'] == opening_action:
                found = i
                break
        return found

    def count_matching(self, symbol):
        return len([1 for x in self.openPositions if x['Symbol'] == symbol])
    
    def calc_profit(self, symbol, pts):
        if symbol == 'MES':
            return pts * 5
        if symbol == 'MNQ':
            return pts * 2
        if symbol == 'MYM':
            return pts * 0.5
        return pts


def print_trade_stats(trades):
    wins = trades.Profit[trades.Profit > 3].count()
    sumWins = trades.Profit[trades.Profit > 3].sum()
    loses = trades.Profit[trades.Profit < -3].count()
    sumLoses = trades.Profit[trades.Profit < -3].sum()
#    durWin = trades.Duration[trades.Profit > 3].sum()
#    durLoss = trades.Duration[trades.Profit < -3].sum()
    winPerc = 100 * wins / (wins + loses)
    avgWin = sumWins / wins
    avgLoss = sumLoses / loses
    ratio = avgWin / -avgLoss
#    avgWinDuration = durWin / wins
#    avgLossDuration = durLoss / loses
    cost = 1.04 * len(trades)

    profit = trades.Profit.sum()
    commisions = trades.Comm.sum()
    print(f'Contracts: {len(trades)} Profit: ${profit - commisions:.2f} Gross: ${profit:.2f} Commisions: ${commisions:.2f}')
    print(f'wins: {wins} loses: {loses} win%: {winPerc:.0f}%')
    print(f'wins: {sumWins} loses: {sumLoses}')
    print(f'avg w: {avgWin:.1f} avg l: {avgLoss:.1f} ratio: {ratio:.1f}')
#    print(f'avg w time: {avgWinDuration:.1f} avg l time: {avgLossDuration:.1f}')

def load_file(fname):
    df = pd.read_csv(f'\\Users\\niroo\\OneDrive\\Documents\\{fname}', usecols=[0,1,2,3,4,5,6], parse_dates={'Timestamp' : [5,6]})
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    return df

def load_fileEx(fname):
    df = pd.read_csv(fname, usecols=[0,1,2,3,4,5,6], parse_dates={'Timestamp' : [5,6]})
    print(f'loaded {fname} {df.shape[0]} {df.shape[1]}')
    print(df)
    return df

def aggregrate_by_sequence(df):
    return df.groupby(['Seq']).agg(
        Symbol=pd.NamedAgg(column="Symbol", aggfunc="first"),
        Action=pd.NamedAgg(column="Action", aggfunc="first"),
        Num=pd.NamedAgg(column="Symbol", aggfunc="count"),
        Profit=pd.NamedAgg(column="Profit", aggfunc="sum") )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process IB trade logs')
    parser.add_argument('--skip', metavar='skip', default=0, type=int, help='number of rows to skip')
    parser.add_argument('--input', metavar='input', default='', help='file name')
    args = parser.parse_args()

    if len(args.input) > 0:
        df = load_fileEx(args.input)
    else:
        df = pd.concat( [load_file('trades-0214.csv'), load_file('trades.20220222.csv'), load_file('trades.20220223.csv')] )

    b = Blotter()
    trades = b.process_trade_log(df, args.skip)
    console.print(trades, style='cyan')
    c = len(b.openPositions)
    if c > 0:
        console.print(f'Open contracts {c}', style='yellow')
    else:
        console.print('All contracts matched', style='yellow')
    console.print(aggregrate_by_sequence(trades), style='cyan')
    print_trade_stats(trades)
