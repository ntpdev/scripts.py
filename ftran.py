#!/usr/bin/python3
fn = 'f:\\a.txt'

with open(fn, 'r') as f:
    text = f.readlines()

a = []
for s in text:
    a.append(s.strip())

a.sort()

with open(fn + '.out', 'w') as fout:
    for s in a:
        fout.write(s + '\n')
    
    x = a[0]
    for s in a[1:]:
        x = x + ',' + s
    fout.write(x + '\n')


