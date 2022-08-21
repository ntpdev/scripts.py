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
    
    first = True
    x = ''
    for s in a:
        if not first:
            x = x + ', '
        x = x + s
        first = False
    fout.write(x + '\n')


