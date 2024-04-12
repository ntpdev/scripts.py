#!/usr/bin/python3
import numpy as np
import pandas as pd
import plotly.express as px
from io import StringIO
import graphviz
from scipy.optimize import brentq
from functools import cache
from datetime import date, timedelta
import math
# import plotly.graph_objs as go
# import plotly.offline as py

def fib(n):
    return n if n < 2 else fib(n-1) + fib(n-2)

def fibm(maxN):
    memo = [-1] * maxN
    memo[0] = 0
    memo[1] = 1

    def fibImpl(n):
        if memo[n] < 0:
            # print(f'calculating {n}')
            memo[n] = fibImpl(n-1) + fibImpl(n-2)
        return memo[n]

    return fibImpl

def fibIter(n):
    '''return a generator iterator'''
    a,b = 0,1
    for _ in range(n):
        yield a
        a,b = b, a+b

# use functools cache decorator
@cache
def fib2(n):
   return n if n < 2 else fib2(n-1) + fib2(n-2)

def collatzIter(n):
    while n > 1:
        yield n
        n = n // 2 if n % 2 == 0 else 3 * n + 1
    yield n

def draw_collatz_seq_length():
    # draw barchart of lengths of various sequences
    d = {x : sum(1 for _ in collatzIter(x)) for x in range(10,100)}
    df = pd.DataFrame.from_dict(d, orient='index')

    fig = px.bar(df, x=df.index, y=df.iloc[:,0], title='Collatz Seq length')
    fig.show()

def draw_gantt_chart():
    s = '''start finish task
    2022-04-01	2023-03-31	£288.71
    2022-11-20	2023-03-31	£69.70
    2022-06-01	2023-03-31	£240.46
    2023-04-01	2024-03-31	£328.87
    2023-04-01	2023-06-06	£39.00
    2022-06-01	2022-11-20	£94.85
    2022-06-01	2023-03-31	-£240.00'''
    df = pd.read_csv(StringIO(s), delim_whitespace=True)
    df.sort_values('start', inplace=True)

    # Create the chart
    fig = px.timeline(df, x_start='start', x_end='finish', y='task', color='task')

    # Customize the chart
    fig.update_layout(title='Gantt Chart', xaxis_title='Date', yaxis_title='Task')

    # Show the chart
    fig.show()

def draw_diag():
    g = graphviz.Graph('G', filename='z.gv', format='svg')
    g.edge('run', 'intr')
    g.edge('intr', 'runbl')
    g.edge('runbl', 'run')
    g.edge('run', 'kernel')
    g.view()

def init_cashflow(drawdown, n, start_date):
  d = start_date.day
  start_date -= timedelta(days=d)
  ix = pd.date_range(start=start_date, periods=n+1 , freq='MS')
  ix = ix + timedelta(days=d-1)
  days = ix.to_series().diff().dt.days.fillna(0).astype(int)
  df = pd.DataFrame({'days' : days, 'capital' : float(drawdown), 'repayment' : 0., 'interest': 0., 'int_paid':0., 'frac_int':0., 'outstanding':0.}, index=ix)
  df.iat[0,6] = df.iat[0,1]
  return df

def evaluate_cashflow(df, repayment, interest_rate):
  '''return the final outstanding amount given a fixed repayment amount'''
  df.iloc[1:,2] = repayment
  prev = None
  for i in df.index:
    # copy prev outstanding to current capital
    if prev:
      df.at[i,'capital'] = df.at[prev,'outstanding']
    # calculate row
    carried_interest = df.at[prev,'frac_int'] if prev else 0.
    df.at[i,'interest'] = df.at[i,'days'] * df.at[i,'capital'] * interest_rate / 365 + carried_interest
    df.at[i,'int_paid'] = math.floor(df.at[i,'interest'] * 100) / 100
    df.at[i,'frac_int'] = df.at[i,'interest'] - df.at[i,'int_paid']
    df.at[i,'outstanding'] = df.at[i,'capital'] - df.at[i,'repayment'] + df.at[i,'int_paid']
    prev = i
  return df.iat[-1,6]

def solve_cashflow(df, interest_rate):
  ubound = df.iat[0,1]
  return brentq(lambda x: evaluate_cashflow(df, x, interest_rate), 0, ubound)

def example_cashflow():
    drawdown = 1200
    interest_rate = .06
    df = init_cashflow(drawdown, 12, date.today())
    r = solve_cashflow(df, interest_rate)
    # re-evaluate with rounded repayment
    evaluate_cashflow(df, round(r,2), interest_rate)
    print(f'drawdown {drawdown:.2f}, interest rate {interest_rate * 100:.4f}%, monthly payment {round(r,2)}')
    print(f'total paid {df["repayment"].sum():.2f} , total interest {df["int_paid"].sum():.2f}, total frac {df["frac_int"].sum()}')
    print(df)

if __name__ == '__main__':
    print([fib(x) for x in range(10)])
    
    f = fibm(20)
    print(f(9))
    print([f(x) for x in range(18)])
    
    print([x for x in fibIter(18)])
    print([x for x in collatzIter(27)])
    # draw_collatz_seq_length()


    xs = [5, -7, 3, 5, 2, -2, 4, -1]
    # replace neg values by 0
    [x if x > 0 else 0 for x in xs]
    # filter out 
    [x for x in xs if x > 0]
    # generator expresions can be used on the fly. will sum the positive integers
    sum(x for x in xs if x > 0)
    # count number positive
    sum(1 for x in xs if x > 0)
    # not as nice using reduce
    #functools.reduce(lambda acc, e : acc + e if e > 0 else acc, xs, 0)

    # draw_gantt_chart()
    #draw_diag()
    example_cashflow()