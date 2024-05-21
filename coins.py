#!/usr/bin/python
from dataclasses import dataclass
from typing import List
from collections import Counter
import unittest

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


def remove_starting(xs):
    n = 0
    e = xs[0]
    while n < len(xs) and e == xs[n]:
        n += 1
    return xs[n:]

# DP solution find the least number of coins for a given total
# minCoins[i] is list of coins needed to make i
# O(c * n) compared to O(c ^ n)
def search(coins, target):
    """
    least number of coins for a given total.
    
    Args:
        coins: List of integers representing different coin values.
        target: An integer representing the total value to be achieved.
        
    Returns:
        A list of integers representing the minimum number of coins needed to make the total.
    """
    minCoins = [[]] * (target + 1)
    for i in range(1, target+1):
        for c in coins:
            if i == c:
                minCoins[i] = [c]
            elif i > c:
                prev = len(minCoins[i-c])
                curr = len(minCoins[i])
                if prev > 0 and (curr == 0 or prev + 1 < curr):
                    minCoins[i] = minCoins[i-c] + [c]

    return minCoins[target]

def coins_at_least_target(coins, target):
    """return list of possible combinations of coins that sum to at least target, in sum order"""
    acc = []  # global list of results

    def search(unused, target, choosen):
        """
        Recursive function that searches for combinations of coins in a wallet to achieve a target value.

        Parameters:
            unused (list): A list of individual coins.
            target (int): The target value to be achieved.
            choosen (list): A list of integers representing the chosen coins in the current combination.

        Returns:
            None. updates acc with possible solutions.
        """
        #print(f'searching {unused} {target} {choosen}')
        if target <= 0:
            acc.append(choosen)
        elif len(unused) > 0:
            c = unused[0]
            # two choices to use c or not use any cs
            search(unused[1:], target-c, choosen + [c])
            search(remove_starting(unused), target, choosen)
                
    search(sorted(coins), target, [])
    acc.sort(key=sum)
    return acc

class TestMethods(unittest.TestCase):
    def test_search(self):
        coins = [1,5,6,8]
        soln = search(coins, 51)
        self.assertEqual(sum(soln), 51)
        self.assertListEqual(soln, [8, 8, 8, 8, 8, 6, 5])
        print(f'min coins {len(soln)} using {soln} for total {sum(soln)}')

    def test_search2(self):
        coins = [5,8]
        soln = search(coins, 51)
        self.assertEqual(sum(soln), 51)
        self.assertListEqual(soln, [8, 8, 5, 5, 5, 5, 5, 5, 5])
        print(f'min coins {len(soln)} using {soln} for total {sum(soln)}')

    def test_search3(self):
        coins = [1,7,10]
        soln = search(coins, 15)
        self.assertEqual(sum(soln), 15)
        self.assertCountEqual(soln, [1, 7, 7])
        print(f'min coins {len(soln)} using {soln} for total {sum(soln)}')

    def test_search_limited_coins_a(self):
        res = coins_at_least_target([1,1,1,1], 3)
        self.assertEqual(len(res), 1)
        self.assertEqual(sum(res[0]), 3)
        self.assertListEqual(res[0], [1, 1, 1])
        print(res)

    def test_search_limited_coins_b(self):
        res = coins_at_least_target([2,2,2,5,10], 6)
        self.assertEqual(len(res), 7)
        self.assertEqual(sum(res[0]), 6)
        self.assertListEqual(res[0], [2, 2, 2])
        print(res)

    def test_search_limited_coins_c(self):
        res = coins_at_least_target([2,5,7], 15)
        self.assertEqual(len(res), 0)
        print(res)

if __name__ == '__main__':
    unittest.main()

