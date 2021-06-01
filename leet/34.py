#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: 34.py
@time: 2021/1/14 20:29
@desc: 
"""
import time


class Solution:
    def searchRange(self, nums, target: int):
        if target not in nums:
            return [-1, -1]
        else:
            res = nums.index(target)
            res_reverse = nums[::-1].index(target)
            return [res, len(nums) - 1 - res_reverse]


if __name__ == '__main__':
    s = Solution()
    nums = [1,2,3]
    target = 2
    print(s.searchRange(nums,target))
