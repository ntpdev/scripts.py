#!/usr/bin/python3
import unittest
import math
import numpy as np
import pandas as pd
import itertools as it
from collections import deque

class Solution:
    def findKlargestEx(self, xs, k):
        return sorted(xs, reverse=True)[k - 1]

    def findKlargest(self, xs, k):
        pivot = xs.pop()
        ls = []
        rs = []
        # partition xs into 2 lists containing elements less than and greater than pivot
        for x in xs:
            (ls if x < pivot else rs).append(x)
        # print(ls,pivot,rs)
        # numLt is the number of elements less than the target
        numLt = k - len(rs) - 1
        if numLt < 0:
            return self.findKlargest(rs, k)
        elif numLt > 0:
            return self.findKlargest(ls, numLt)
        return pivot

    def largestValueLessOrEqualEx(self, xs, k):
        closest = -1
        for x in xs:
            if x > k:
                return closest
            closest = x
        return closest

# find value closest but not greater than k in the sorted list xs
    def largestValueLessOrEqual(self, xs, k):
        closest = -1
        l = 0
        r = len(xs) - 1
        closest = -1
        while l <= r:
            m = (l + r) // 2
            x = xs[m]
            if x == k:
                return k
            if x < k:
                l = m + 1
                closest = x
            else: # x > k
                r = m - 1
        return closest


    # return True if xs contains k
    def containsBinarySearch(self, xs, k):
        l = 0
        r = len(xs) - 1
        while l <= r:
            m = (r + l) // 2
            x = xs[m]
            if x == k:
                return True
            if x < k:
                l = m + 1
            else:
                r = m - 1
        return False

# return size of smallest subarray that sums to at least k where all elements of xs are positive
    def minSizeSubarraySum(self, xs, k):
        l = 0
        windowSum = 0
        notfound = len(xs) + 1
        sz = notfound
        for r in range(len(xs)):
            windowSum += xs[r]
            while windowSum >= k:
                sz = min(sz, r - l + 1)
                windowSum -= xs[l]
                l += 1 # this can advance l beyond r but in that case windowSum will be 0
                
        return sz if sz < notfound else 0
    
    def repeatedSubstringOfLength(self, xs, n):
        # dict of string -> count
        seqs = {}
        for i in range(len(xs) - n + 1):
            s = xs[i:i+n]
            c = seqs.get(s, 0)
            seqs[s] = c + 1
        return [k for k,v in seqs.items() if v > 1]
    
    # return length of longest substring without dup characters
    # sliding window solutions can either move the right or left end
    # there is state associated with the window and the algorithm needs to update the state
    # depending on whether the left or right is moved. Updating the state is normally O(1)
    # compared to recalculating the state which is O(n) hence the improvement
    def longestSubstringNoDup(self, s):
        longest = 0
        p = 0 # index of start of window
        chars = set() # state of window is the unique characters in the window
        for r,c in enumerate(s):
            if c in chars:
                removed = False
                while not removed:
                    x = s[p]
                    # print('removing', r, chars)
                    chars.remove(x)
                    p += 1
                    removed = x == c
            # print('adding', p,r,c, longest)
            longest = max(longest, r - p + 1)
            chars.add(c)
        return longest
    
    # 
    def minWindowSubstring(self, s, t):
        """minimum length substring of s that contains all the characters in t or '' if none"""
        ts = set(t)
        window = LetterFreqency() # dict of char -> count for chars if interest in window
        res = None
        p = 0
        for r,c in enumerate(s):
            # print(p, r, c, window, res)
            if c in ts:
                window.add(c)
                while window.unique() == len(ts):
                    # print('soln window', res, s[p:r+1])
                    if res is None or r - p + 1 < len(res):
                        res = s[p:r+1]
                    # shink left, removing char from window if it is a char we need to track
                    left = s[p]
                    p += 1
                    if left in ts:
                        window.remove(left)
                
        return res if res is not None else ''
    
    def longestRepeatingCharReplacement(self,s,k):
        """longest substring of a single char after making at most k replacements"""
        mx = 0
        p = 0
        window = LetterFreqency()
        for r,c in enumerate(s):
            window.add(c)
            # print(s[p:r+1], window, mx)
            if window.total() - window.maxFrequency() <= k:
                # valid window
                mx = max(mx, r - p + 1)
            else:
                # invalid window - shrink until valid
                while window.total() - window.maxFrequency() > k:
                    window.remove(s[p])
                    p += 1
        return mx
    
    def maxSlidingWindowEx(self, nums, k):
        out = []
        for l in range(len(nums) - k + 1):
            out.append(max(nums[l:l+k]))
        return out

    # the optimised solution uses a queue to hold elements in the window
    # which can become the max. Elements in the queue smaller than the new addition
    # can be removed from the queue as they can never become the max. The left element in
    # the queue is removed if it is the element being dropped from the window
    def maxSlidingWindow(self, nums, k):
        out = []
        q = deque() # stores the elements that could become the max in the window
        for r in range(len(nums)):
            l = r - k # l is index to drop from window, r is index to add
            # print(nums, 0 if l < 0 else nums[l],nums[r])
            # remove left item from queue if it is being dropped from window
            if l >= 0 and nums[l] == q[0]:
                q.popleft()
            right = nums[r]
            # remove items in queue smaller than elem added
            while q and right > q[-1]:
                q.pop() # remove last item
            q.append(right)
            # print(q)
            if r >= k - 1:
                out.append(q[0]) # current max in window is always idx 0
        return out
    
    def twoSumEx(self, nums, k):
        """find index of elements in nums that sum to k"""
        for x in range(len(nums) - 1):
            for y in range(x+1, len(nums)):
                print(x,y)
                if nums[x] + nums[y] == k:
                    return x,y
        return 0,0

    def twoSum(self, nums, k):
        """find index of elements in nums that sum to k"""
        d = {} # dict of value -> index
        for i,n in enumerate(nums):
            if k - n in d:
                return d[k-n],i
            d[n] = i
        return 0,0
    
    def twoSumSorted(self, nums, k):
        """find index of pair of elements in nums that sum to k. nums is sorted"""
        l = 0
        r = len(nums)-1
        while l < r:
            diff = nums[l] + nums[r] - k
            if diff == 0:
                return l+1, r+1
            if diff < 0:
                l += 1
            else:
                r -= 1
        return 0,0
    
    def threeSumToZero(self, nums):
        """return unique triplets that sum to 0. This should not contain duplicates"""
        # this might not be quite right
        xs = sorted(nums)
        out = []
        for i,n in enumerate(xs):
            if n > 0:
                return out
            l = i + 1
            r = len(xs) - 1
            # now 2 sum for -n
            while l < r:
                sum = xs[l] + xs[r] + n
                if sum == 0:
                    out.append([n, xs[l], xs[r]])
                    break # only find 1 solution using n. is this correct?
                elif sum < 0:
                    l += 1
                else:
                    r -= 1
        return []
    
    def numberCombinationsGt(self, xs, ys, bound):
        '''return number of combinations which sum to greater than bound'''
        # use list comp with nested generators and a condition
        zs = [(x,y) for x in xs for y in ys if (x+y) > bound]
        # alternatively using itertools
        # [e for e in itertools.product(xs, ys) if sum(e) > 9]
        print(f'{len(zs)}/{len(xs) * len(ys)} {zs}')
        return len(zs)

    def numberCombinationsGtEx(self, xs, ys, bound):
        '''return number of combinations which sum to greater than bound'''
        # Use itertools.product to create cartesian product then * starred expression to unpack the tuples
        x2, y2 = zip(*(it.product(xs, ys)))
        df = pd.DataFrame({'x':x2, 'y':y2})
        df['c'] = df.x + df.y
        r = df[df.c > bound]
        print(f'{r}')
        return len(r)
    
    def primesUpTo(self, n):
        '''return a np array containing all primes up to n'''
        sieve = np.arange(n)
        sieve[:2] = 0
        for i in range(2, math.isqrt(n) + 1):
            if sieve[i]:
                sieve[i*i::i] = 0
        return sieve.nonzero()[0]


# python has a Counter class which is a specialised dict for counting
class LetterFreqency():
    def __init__(self):
        self.counts = [0] * 26
    
    def __str__(self):
        return str(self.asDictionary())
    
    def __eq__(self, other):
        for x,y in zip(self.counts, other.counts):
            if x != y:
                return False
        return True
        
    def add(self, s):
        r = 0
        for ch in s:
            x = self.indexOf(ch)
            r = self.counts[x] + 1
            self.counts[x] = r
        return r
    
    def remove(self, s):
        r = 0
        for ch in s:
            x = self.indexOf(ch)
            r = self.counts[x]
            if r == 0:
                raise ValueError(f'No count recorded for character {ch}')
            r -= 1
            self.counts[x] = r
        return r
    
    def unique(self):
        return len([e for e in self.counts if e > 0])
    
    def total(self):
        return sum(self.counts)

    def frequency(self, ch):
        return self.counts[self.indexOf(ch)]
    
    def maxFrequency(self):
        return max(self.counts)
    
    def indexOf(self, ch):
        x = ord(ch) - ord('a')
        if x < 0 or x > 25:
            raise IndexError(f'Character {ch} out of range')
        return x
    
    def asDictionary(self):
        return {(chr(k + ord('a')),v) for k,v in enumerate(self.counts) if v > 0}
    
    
class TestMethods(unittest.TestCase):
    def test_findKLargest(self):
        xs = [3,2,1,5,6,4]
        k = 2
        self.assertEqual(Solution().findKlargest(xs, k), 5)

    def test_findLarestLessThanEqual(self):
        xs = [x * 5 for x in range(256)]
        k = 323
        self.assertEqual(Solution().largestValueLessOrEqual(xs, k), 320)

    def test_minSSA(self):
        self.assertEqual(Solution().minSizeSubarraySum([1], 2), 0)
        self.assertEqual(Solution().minSizeSubarraySum([1,1,1], 2), 2)
        self.assertEqual(Solution().minSizeSubarraySum([2,3,1], 4), 2)
        self.assertEqual(Solution().minSizeSubarraySum([1,1,1,5,1,1,4,1], 4), 1)
        self.assertEqual(Solution().minSizeSubarraySum([1,2,3,6], 6), 1)
        self.assertEqual(Solution().minSizeSubarraySum([1,4,4], 4), 1)
        self.assertEqual(Solution().minSizeSubarraySum([2,3,1,2,4,3], 7), 2)

    def test_repeatedSubstringOfLength(self):
        self.assertEqual(Solution().repeatedSubstringOfLength('ABCA', 2), [])
        self.assertEqual(Solution().repeatedSubstringOfLength('ABCABA', 2), ['AB'])
        self.assertEqual(Solution().repeatedSubstringOfLength('AACCAABBABCCAAB', 4), ['CCAA', 'CAAB'])

    def test_longestSubstringNoDup(self):
        self.assertEqual(Solution().longestSubstringNoDup('a'), 1)
        self.assertEqual(Solution().longestSubstringNoDup('abc'), 3)
        self.assertEqual(Solution().longestSubstringNoDup('abb'), 2)
        self.assertEqual(Solution().longestSubstringNoDup('abcbef'), 4)
        self.assertEqual(Solution().longestSubstringNoDup('abcabcbb'), 3)
    
    def test_minWindowSubstring(self):
        self.assertEqual(Solution().minWindowSubstring('abcd', 'ef'), '')
        self.assertEqual(Solution().minWindowSubstring('abcdb', 'db'), 'db')
        self.assertEqual(Solution().minWindowSubstring('abbabcab', 'ac'), 'ca')
        self.assertEqual(Solution().minWindowSubstring('adobecodebanc', 'abc'), 'banc')

    def test_longestRepeatingCharReplacement(self):
        self.assertEqual(Solution().longestRepeatingCharReplacement('abab', 2), 4)
        self.assertEqual(Solution().longestRepeatingCharReplacement('aabababbba', 1), 5)
    
    def test_maxSlidingWindow(self):
        self.assertEqual(Solution().maxSlidingWindowEx([1,3,-1,-3,5,4,6,7], 3), [3,3,5,5,6,7])
        self.assertEqual(Solution().maxSlidingWindow([1,3,-1,-3,5,4,6,7], 3), [3,3,5,5,6,7])
        self.assertEqual(Solution().maxSlidingWindow([3,5,5,5,8,5,3,3,3], 3), [5,5,8,8,8,5,3])
    
    def test_twoSum(self):
        self.assertEqual(Solution().twoSum([1,3,2,4], 6), (2,3))

    def test_twoSumSorted(self):
        self.assertEqual(Solution().twoSumSorted([2,7,11,15], 9), (1,2))
        self.assertEqual(Solution().twoSumSorted([1,3,4,5,7,10,11], 9), (3,4))

    def test_threeSumToZero(self):
        self.assertEqual(Solution().threeSumToZero([-1, 0, 1, 2, -1, -4]),[[-1, -1, 2], [-1, 0, 1]])
        self.assertEqual(Solution().threeSumToZero([-3,3,4,-7,1,2]), [[-7,3,4],[-3,1,2]])

    def test_queue(self):
        q = deque()
        q.append(8)
        q.append(3)
        q.append(1)
        n = 5
        while q and q[-1] < n:
            q.pop()
        q.append(n)
    
    def test_LetterFreqency(self):
        d = LetterFreqency()
        self.assertEqual(d.indexOf('a'), 0)
        self.assertEqual(d.indexOf('z'), 25)
        self.assertEqual(d.add('a'), 1)
        self.assertEqual(d.add('a'), 2)
        self.assertEqual(d.remove('a'), 1)
        self.assertEqual(d.remove('a'), 0)
        d.add('mississippi')
        self.assertEqual(d.unique(), 4)
        self.assertEqual(d.total(), 11)
        d.remove('issss')
        self.assertEqual(d.unique(), 3)
        self.assertEqual(d.total(), 6)
        self.assertEqual(d.frequency('i'), 3)
        self.assertEqual(d.frequency('s'), 0)
        x = LetterFreqency()
        y = LetterFreqency()
        self.assertTrue(x == y)
        x.add('hello')
        self.assertFalse(x == y)
        y.add('eholl')
        self.assertTrue(x == y)

    def test_numberCombinationsGt(self):
        xs = [1,1,2,3,4,5]
        ys = [2,3,4,5,6,6]
        r = Solution().numberCombinationsGt(xs, ys, 9)
        self.assertTrue(r == 5)
        r = Solution().numberCombinationsGtEx(xs, ys, 9)
        self.assertTrue(r == 5)
    
    def test_primesUpTo(self):
        # 104729 is 10,000th prime
        r = Solution().primesUpTo(104730)
        self.assertEqual(len(r), 10000)
        self.assertEqual(r[-1], 104729)

if __name__ == '__main__':
    unittest.main()