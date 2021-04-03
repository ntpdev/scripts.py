#!/usr/bin/python

coins = [1,5,6,8]
target = 51

def cons(e, xs):
    ys = xs.copy()
    ys.append(e)
    return ys

# DP solution find the least number of coins for a given total
# minCoins[i] is list of coins needed to make i
# O(c * y) compared to O(c ^ n)
def search():
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

soln = search()
print(f'min coins {len(soln)}')
print(soln)
