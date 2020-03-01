#!/usr/bin/python

def distribute(initial, people):
    remaining = initial
    xs = [0] * people
    n = 1
    while remaining > 0:
        x = min(n, remaining)
        xs[(n-1) % len(xs)] += x
        remaining -= x
        n = n + 1
    assert sum(xs) == initial
    return xs

# given a set of student ratings allocate candies such that
# - each student has at least 1
# - a student has more than his immediate neighbours if his score is higher
# xs[i] is candies for student i
def soln(ratings):
    n = len(ratings)
    xs = [1] * n
    for i in range(1,n):
        print(f'r {i} {ratings[i]}')
        if ratings[i] > ratings[i-1] and xs[i] <= xs[i-1]:
            xs[i] = xs[i-1] + 1
        elif ratings[i] == ratings[i-1] and xs[i] < xs[i-1]:
            xs[i] = xs[i-1]
    for i in range(n-2,-1,-1):
        print(f'l {i} {ratings[i]}')
        if ratings[i] > ratings[i+1] and xs[i] <= xs[i+1]:
            xs[i] = xs[i+1] + 1
        elif ratings[i] == ratings[i+1] and xs[i] < xs[i+1]:
            xs[i] = xs[i+1]
    print(xs)
    return sum(xs)

ratings = [3,3,2,1,2,3,3]
print(soln([2,2,2,2]))
print(soln([2,2,4,6,8,8]))
print(soln([9,9,7,5,3,3]))
print(soln(ratings))

print(distribute(1000,10))
