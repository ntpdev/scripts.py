xs = [10,15,20]
xs = [1,100,1,1,1,100,1,1,100,1]
ys = {0 : xs[0], 1 : xs[1]}

def mincost(n):
    if n in ys:
        return ys[n]

    c = xs[n] if n < len(xs) else 0
    c += min(mincost(n-1), mincost(n-2))

    ys[n] = c
    print(ys)
    return c

print(f'mincost {mincost(len(xs))}')