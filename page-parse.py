from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import re

# parse string 'Apple - NASDAQ:AAPL\nAAPL'
# 1 AAPL
# 2 NASDAQ:AAPL
# 3 AAPL
pattern = re.compile('(.+).-.(.+)\\n(\w+)')

# ['1', 'Apple - NASDAQ:AAPL\nAAPL', 'AAPL', '313']
def parseRow(cols):
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

fn = 'c:\\users\\niroo\\downloads\\Motley Fool Disclosure.html'

with open(fn, 'r') as f:
    html_doc = f.read()

soup = BeautifulSoup(html_doc, 'html.parser')

#table = soup.find('table', attrs={'class':'lineItemsTable'})
table = soup.find('table')
table_body = table.find('tbody')

data = []
rows = table_body.find_all('tr')
for row in rows:
    cols = row.find_all('td')
    cols = [ele.text.strip() for ele in cols]
    data.append(parseRow(cols))
#    data.append([ele for ele in cols if ele]) # Get rid of empty values

df = pd.DataFrame(data)
print(df)
df.to_csv('c:\\users\\niroo\\downloads\\Motley Fool Disclosure.csv', index=False)