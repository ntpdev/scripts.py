#!/usr/bin/python3

def fibm(maxN):
    memo = [-1] * maxN
    memo[0] = 0
    memo[1] = 1

    def fibImpl(n):
        if memo[n] < 0:
            memo[n] = fibImpl(n-1) + fibImpl(n-2)
        return memo[n]

    return lambda n : fibImpl(n)

def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print([fib(x) for x in range(10)])

f = fibm(20)
print(f(9))
print([f(x) for x in range(18)])
