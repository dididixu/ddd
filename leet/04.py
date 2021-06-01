#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: 04.py
@time: 2020/11/30 10:14
@desc: 
"""


class Solution:
    def findNumberIn2DArray(self, matrix, target: int) -> bool:
        if (matrix == None or len(matrix) == 0):
            return False
        n, m = len(matrix), len(matrix[0])
        row, col = 0, m - 1
        while(row < n and col >= 0):
            if matrix[row][col]==target:
                return True
            elif matrix[row][col] < target:
                row += 1
            elif matrix[row][col] > target:
                col -= 1
        return False

if __name__ == "__main__":
    mat = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30]
    ]
    s = Solution()
    print(s.findNumberIn2DArray(mat, 20))
