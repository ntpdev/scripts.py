from bs4 import BeautifulSoup
from datetime import date
import requests
import pandas as pd
import re

# extract table of stocks from motley fool disclosure page
# using requests http library https://requests.readthedocs.io/en/latest/
# and BeautifulSoup html parsing library https://beautiful-soup-4.readthedocs.io/en/latest/index.html

# parse string 'Apple - NASDAQ:AAPL\nAAPL'
# 1 AAPL
# 2 NASDAQ:AAPL
# 3 AAPL
pattern = re.compile('(.+).-.(.+)\\n(\w+)')

# ['1', 'Apple - NASDAQ:AAPL\nAAPL', 'AAPL', '313']
def parse_fool_disclosure(cols):
    m = pattern.match(cols[1])
    d = {}
    d['rank'] = cols[0]
    if m:
        d['company'] = m.group(1)
        d['exchange'] = m.group(2)
    else:
        d['company'] = cols[1]
        d['exchange'] = 'NA'
    d['ticker'] = cols[2]
    d['held'] = cols[3]
    return d

def extract_fool_stocks(soup):
    table = soup.find('table')
    table_body = table.find('tbody')

    data = []
    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append(parse_fool_disclosure(cols))
#        data.append([ele for ele in cols if ele]) # Get rid of empty values
    return pd.DataFrame(data)

def download_fool_disclosure(url, fn):
    print('loading ' + url)
    r = requests.get(url)
    if not r.ok:
        print('http status ' + r.status_code)
        raise Exception(f'failed to load page {url}')
    soup = BeautifulSoup(r.text, 'html.parser')
    df = extract_fool_stocks(soup)
    print(df)
    df.to_csv(fn, index=False)
    print('saved ' + fn)


fn = f'c:\\users\\niroo\\downloads\\Motley Fool Disclosure {date.today().isoformat()}.csv'
df = download_fool_disclosure('https://www.fool.com/legal/fool-disclosure/', fn)
