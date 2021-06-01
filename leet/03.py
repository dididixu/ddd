#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: 03.py
@time: 2020/11/30 9:55
@desc: 
"""


class Solution:
    def findRepeatNumber(self, nums) -> int:
        nums = sorted(nums, reverse=False)
        print(nums)
        for i in range(len(nums) - 1):
            if nums[i] == nums[i + 1]:
                return nums[i]
        return 0


if __name__ == "__main__":
    s = Solution()
    print(s.findRepeatNumber([2, 3, 1, 0, 2, 5, 3]))
