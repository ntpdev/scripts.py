#!/usr/bin/python

openbrace = '([{'
closebrace = ')]}'

def isopen(ch):
  return ch in openbrace
  
def isclose(ch):
  return ch in closebrace

def matches(a,b):
  return openbrace.find(a) == closebrace.find(b)

def matchBrackets(s):
    stack = []
    for c in s:
        if isopen(c):
            stack.append(c)
        elif isclose(c):
            if len(stack) > 0 and matches(stack[-1], c):
                stack.pop()
            else:
                print(f'{c} does not match {stack}')
                return False
    print(stack)
    return len(stack) == 0

assert matchBrackets('') == True
assert matchBrackets('()') == True
assert matchBrackets('(') == False
assert matchBrackets(')') == False
assert matchBrackets('[]') == True
assert matchBrackets('[{}]') == True
assert matchBrackets('[]{(})') == True
assert matchBrackets('[((a)(b))]') == True
