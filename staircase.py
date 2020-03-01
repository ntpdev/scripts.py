#!/usr/bin/python

# return number of ways to climb a staircase of steps-1 
# with possible steps of size sz
# this is a generative solution (this problem is a variation of the fibonacci)
def numWays(steps, sz):
    ways = [0] * steps
    ways[0] = 1
    for i in range(steps):
        for s in sz:
            if i + s < steps:
                ways[i+s] += ways[i]
                print(ways)
    return ways[-1]

print(numWays(3, [1,2]))

print(numWays(8, [1,2]))

#print(numWays(21, [1,3,7,15]))