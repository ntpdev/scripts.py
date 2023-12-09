#!/usr/bin/python
from typing import List

coins = [1,5,6,8]

def isNotMultiple(n):
    return lambda e : e % n != 0

def andCombinator(f, g):
    return lambda e : f(e) and g(e)

def filter(n):
    return lambda xs : [e for e in xs if e % n != 0]

def xfilter(xs):
    if (len(xs)) == 1:
        return xs
    print(type(xs))
    return xs[:1] + xfilter(filter(xs[0])(xs[1:]))

def primes(n):
    return sieve(range(2,n+1), [])

def sieve(xs, acc) -> List[int]:
    if len(xs) == 1:
        acc.append(xs[0])
        return acc
    h = xs[0] # once h is > sqrt(n) all remaining elements are prime
    acc.append(h)
    return sieve([e for e in xs[1:] if e % h != 0], acc)

def isNotMultipleAll(xs):
    f = None
    for x in xs:
        g = lambda e : e % x != 0
        f = andCombinator(f,g) if f else g
    return f

def cons(e, xs):
    ys = xs.copy()
    ys.append(e)
    return ys

# DP solution find the least number of coins for a given total
# minCoins[i] is list of coins needed to make i
# O(c * n) compared to O(c ^ n)
def search(target):
    minCoins = [[]] * (target + 1)
    for i in range(1, target+1):
        for c in coins:
            if i == c:
                minCoins[i] = [c]
            elif i > c:
                prev = len(minCoins[i-c])
                curr = len(minCoins[i])
                if prev > 0 and (curr == 0 or prev + 1 < curr):
                    minCoins[i] = cons(c, minCoins[i-c])
        # print(minCoins)
    return minCoins[target]

soln = search(51)
print(f'min coins {len(soln)} using {soln} for total {sum(soln)}')

soln = primes(256)
print(soln)

