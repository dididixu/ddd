#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: 07-前序中序给出二叉树.py
@time: 2020/12/2 10:41
@desc: 
"""


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def fun(self, preorder, inorder, node: TreeNode, flag):
        if len(preorder) == 0 or len(inorder) == 0:
            return
        if preorder[0] in inorder:
            index = inorder.index(preorder[0])
            node_new = TreeNode(preorder[0])
            if flag == 0:
                node.left = node_new
            elif flag == 1:
                node.right = node_new
            self.fun(preorder[1:index + 1], inorder[0:index], node_new, 0)
            self.fun(preorder[index + 1:], inorder[index + 1:], node_new, 1)
        else:
            return

    #
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if len(preorder) == 0 or len(inorder) == 0:
            return
        node = TreeNode(preorder[0])
        index = inorder.index(preorder[0])
        self.fun(preorder[1:index + 1], inorder[0:index], node, 0)
        self.fun(preorder[index + 1:], inorder[index + 1:], node, 1)
        return node


if __name__ == "__main__":
    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    s = Solution()
    node = s.buildTree(preorder, inorder)
    print(node.val)
    node_l = node
