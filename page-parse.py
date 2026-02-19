#!/usr/bin/python3
from bs4 import BeautifulSoup
from datetime import date
import httpx
import pandas as pd
import re
import itertools
from pathlib import Path

# extract table of stocks from motley fool disclosure page
# using httpx http library https://www.python-httpx.org/ with HTTP/2 support
# and BeautifulSoup html parsing library https://beautiful-soup-4.readthedocs.io/en/latest/index.html

# parse string 'Apple - NASDAQ:AAPL\nAAPL'
# 1 AAPL
# 2 NASDAQ:AAPL
# 3 AAPL
pattern = re.compile("(.+).-.(.+)\\n(\\w+)")


# ['1', 'Apple - NASDAQ:AAPL\nAAPL', 'AAPL', '313']
def parse_fool_disclosure(cols):
    m = pattern.match(cols[1])
    d = {}
    d["rank"] = cols[0]
    if m:
        d["company"] = m.group(1)
        d["exchange"] = m.group(2)
    else:
        d["company"] = cols[1]
        d["exchange"] = "NA"
    d["ticker"] = cols[2]
    d["held"] = cols[3]
    return d


def extract_fool_stocks(soup):
    table = soup.find("table")
    table_body = table.find("tbody")

    data = []
    rows = table_body.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        data.append(parse_fool_disclosure(cols))
    #        data.append([ele for ele in cols if ele]) # Get rid of empty values
    return pd.DataFrame(data)


def download_fool_disclosure(url, fn):
    print("loading " + url)
    with httpx.Client(http2=True) as client:
        r = client.get(url)
    if not r.is_success:
        print("http status " + str(r.status_code))
        raise Exception(f"failed to load page {url}")
    soup = BeautifulSoup(r.text, "html.parser")
    df = extract_fool_stocks(soup)
    print(df)
    df.to_csv(fn, index=False)
    print("saved " + str(fn))


def print_changes():
    fn = Path.home() / "Downloads"
    paths = [s for s in fn.glob("Motley Fool Disclosure*.csv")]
    # load each csv into a df and index by date
    # df has cols rank and ticker and makes the ticker the index
    ds = {
        date.fromisoformat(str(f)[-14:-4]): pd.read_csv(f, usecols=["rank", "ticker"])
        for f in paths
    }
    for k, v in ds.items():
        v.set_index("ticker", inplace=True)
        v.rename(columns={"rank": k}, inplace=True)
    # concat the df's into a single df using the date
    df = pd.concat(ds.values(), axis=1, keys=ds)
    # add an action col indicating the change between the first and last rank
    a = {e: "drop" for e in df[df.iloc[:, -1].isna()].index}
    b = {e: "new" for e in df[df.iloc[:, 0].isna()].index}
    c = {e: "up" for e in df[df.iloc[:, 0] > df.iloc[:, -1]].index}
    d = {e: "down" for e in df[df.iloc[:, 0] < df.iloc[:, -1]].index}
    m = {k: v for k, v in itertools.chain(a.items(), b.items(), c.items(), d.items())}
    df["action"] = pd.Series(m)
    df["action"] = df["action"].fillna(".")
    print(df[:20])


def main():
    name = f"Motley Fool Disclosure {date.today().isoformat()}.csv"
    fn = Path.home() / "Downloads" / name
    return download_fool_disclosure("https://www.fool.com/legal/fool-disclosure/", fn)


if __name__ == "__main__":
    main()
    # print_changes()
