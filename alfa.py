#!/usr/bin/python

# returns max num of chars of b that overlap
#
def overlap(a, b):
#  print('---')
  for i in range(len(a)):
    n = min(len(a)-i, len(b))
    #print(f'{n} {a[i:n+i]} {b[0:n]}')
    if a[i:n+i] == b[0:n]:
#      print(f'overlap {n} {a[i:n+i]} {b[0:n]}')
      if len(b[n:]) > len(a[n+i:]):
        print(a[0:n+i] + b[n:])
      else:
        print(a[0:n+i] + a[n+i:])
      return n    
  return 0

def maxOverlap(a,b):
  return max(overlap(a,b), overlap(b,a))

s = 'abcdef'
xs = 'O draconia;conian devil! Oh la;h lame sa;saint!'.split(';')
#['O draconia','conian devil! Oh la','h lame sa','saint!']''
xs = 'm quaerat voluptatem.;pora incidunt ut labore et d;, consectetur, adipisci velit;olore magnam aliqua;idunt ut labore et dolore magn;uptatem.;i dolorem ipsum qu;iquam quaerat vol;psum quia dolor sit amet, consectetur, a;ia dolor sit amet, conse;squam est, qui do;Neque porro quisquam est, qu;aerat voluptatem.;m eius modi tem;Neque porro qui;, sed quia non numquam ei;lorem ipsum quia dolor sit amet;ctetur, adipisci velit, sed quia non numq;unt ut labore et dolore magnam aliquam qu;dipisci velit, sed quia non numqua;us modi tempora incid;Neque porro quisquam est, qui dolorem i;uam eius modi tem;pora inc;am al'.split(';')
assert(overlap(s, 'abc') == 3)
assert(overlap('abc', s) == 3)
assert(overlap(s, 'defg') == 3)
assert(overlap('defg', s) == 0)
assert(overlap(s, 'xyzabc') == 0)
assert(overlap('xyzabc', s) == 3)
assert(overlap(s,'bcde') == 4)
assert(overlap('bcde', s) == 0)
assert(overlap(s,'xcdez') == 0)
assert(overlap('xcdez', s) == 0)
assert(overlap('abab', 'abc') == 2)
assert(maxOverlap(s, 'xyzabc') == 3)

print(f'fragments {len(xs)}')
for a in range(len(xs)):
  for b in range(len(xs)):
    if a != b:
      overlap(xs[a], xs[b])

