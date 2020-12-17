from typing import List, Optional
from collections import defaultdict


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


# 递归
# 时间复杂度：O(n)，n为节点数，访问每个节点恰好一次。
# 空间复杂度：空间复杂度：O(h)，h为树的高度。最坏情况下需要空间O(n)，平均情况为O(logn)

# 递归1：二叉树遍历最易理解和实现版本
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        # 前序递归
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)
        # # 中序递归
        # return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
        # # 后序递归
        # return self.postorderTraversal(root.left) + self.postorderTraversal(root.right) + [root.val]

    # 递归2：通用模板，可以适应不同的题目，添加参数、增加返回条件、修改进入递归条件、自定义返回值


class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        def dfs(cur):
            if not cur:
                return
                # 前序递归
            res.append(cur.val)
            dfs(cur.left)
            dfs(cur.right)
            # # 中序递归
            # dfs(cur.left)
            # res.append(cur.val)
            # dfs(cur.right)
            # # 后序递归
            # dfs(cur.left)
            # dfs(cur.right)
            # res.append(cur.val)

        res = []
        dfs(root)
        return res


# 迭代
# 时间复杂度：O(n)，n为节点数，访问每个节点恰好一次。
# 空间复杂度：O(h)，h为树的高度。取决于树的结构，最坏情况存储整棵树，即O(n)

# 迭代1：前序遍历最常用模板（后序同样可以用）
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        res = []
        stack = [root]  # 循环条件为栈空，因此需要先放入根节点
        # # 前序迭代模板
        while stack:
            cur = stack.pop()
            res.append(cur.val)  # 前序
            # 进栈元素必须不为空，初始得根节点也是判断不为空后加入的
            if cur.right:  # 进栈顺序先右后左（保证出栈顺序为先左后右）
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

        # # 后序迭代，相同模板：将前序迭代进栈顺序稍作修改，最后得到的结果反转
        # while stack:
        #     cur = stack.pop()
        #     if cur.left:
        #         stack.append(cur.left)
        #     if cur.right:
        #         stack.append(cur.right)
        #     res.append(cur.val)  # 后序
        # return res[::-1]


# 迭代1：层序遍历最常用模板
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []
        cur, res = [root], []
        while cur:
            lay, layval = [], []
            for node in cur:
                layval.append(node.val)
                if node.left:
                    lay.append(node.left)
                if node.right:
                    lay.append(node.right)
            cur = lay
            res.append(layval)
        return res


# 迭代2：前、中、后序遍历通用模板（只需一个栈的空间）
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = []
        cur = root
        # 中序，模板：先用指针找到每颗子树的最左下角，然后进行进出栈操作
        while stack or cur:
            while cur:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
        return res

        # # 前序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.left
        #     cur = stack.pop()
        #     cur = cur.right
        # return res

        # # 后序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.right
        #     cur = stack.pop()
        #     cur = cur.left
        # return res[::-1]


# 迭代3：标记法迭代（需要双倍的空间来存储访问状态）：
# 前、中、后、层序通用模板，只需改变进栈顺序或即可实现前后中序遍历，
# 而层序遍历则使用队列先进先出。0表示当前未访问，1表示已访问。
class Solution:
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        stack = [(0, root)]
        while stack:
            flag, cur = stack.pop()
            if not cur:
                continue
            if flag == 0:
                # 前序，标记法
                stack.append((0, cur.right))
                stack.append((0, cur.left))
                stack.append((1, cur))

                # # 后序，标记法
                # stack.append((1, cur))
                # stack.append((0, cur.right))
                # stack.append((0, cur.left))

                # # 中序，标记法
                # stack.append((0, cur.right))
                # stack.append((1, cur))
                # stack.append((0, cur.left))
            else:
                res.append(cur.val)
        return res

        # # 层序，标记法
        # res = []
        # queue = [(0, root)]
        # while queue:
        #     flag, cur = queue.pop(0)  # 注意是队列，先进先出
        #     if not cur: continue
        #     if flag == 0:
        # 层序遍历这三个的顺序无所谓，因为是队列，只弹出队首元素
        #         queue.append((1, cur))
        #         queue.append((0, cur.left))
        #         queue.append((0, cur.right))
        #     else:
        #         res.append(cur.val)
        # return res


# 莫里斯遍历
# 时间复杂度：O(n)，n为节点数，看似超过O(n)，有的节点可能要访问两次，实际分析还是O(n)，具体参考大佬博客的分析。
# 空间复杂度：O(1)，如果在遍历过程中就输出节点值，则只需常数空间就能得到中序遍历结果，空间只需两个指针。
# 如果将结果储存最后输出，则空间复杂度还是O(n)。

# PS：莫里斯遍历实际上是在原有二叉树的结构基础上，构造了线索二叉树，
# 线索二叉树定义为：原本为空的右子节点指向了中序遍历顺序之后的那个节点，把所有原本为空的左子节点都指向了中序遍历之前的那个节点
# emmmm，好像大学教材学过，还考过

# 此处只给出中序遍历，前序遍历只需修改输出顺序即可
# 而后序遍历，由于遍历是从根开始的，而线索二叉树是将为空的左右子节点连接到相应的顺序上，使其能够按照相应准则输出
# 但是后序遍历的根节点却已经没有额外的空间来标记自己下一个应该访问的节点，
# 所以这里需要建立一个临时节点dump，令其左孩子是root。并且还需要一个子过程，就是倒序输出某两个节点之间路径上的各个节点。
# 具体参考大佬博客

# 莫里斯遍历，借助线索二叉树中序遍历（附前序遍历）
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        # cur = pre = TreeNode(None)
        cur = root

        while cur:
            if not cur.left:
                res.append(cur.val)
                # print(cur.val)
                cur = cur.right
            else:
                pre = cur.left
                while pre.right and pre.right != cur:
                    pre = pre.right
                if not pre.right:
                    # print(cur.val) 这里是前序遍历的代码，前序与中序的唯一差别，只是输出顺序不同
                    pre.right = cur
                    cur = cur.left
                else:
                    pre.right = None
                    res.append(cur.val)
                    # print(cur.val)
                    cur = cur.right
        return res


# N叉树遍历
# 时间复杂度：时间复杂度：O(M)，其中 M 是 N 叉树中的节点个数。每个节点只会入栈和出栈各一次。
# 空间复杂度：O(M)。在最坏的情况下，这棵 N 叉树只有 2 层，所有第 2 层的节点都是根节点的孩子。
# 将根节点推出栈后，需要将这些节点都放入栈，共有 M−1个节点，因此栈的大小为 O(M)。


"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


# N叉树简洁递归
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root: return []
        res = [root.val]
        for node in root.children:
            res.extend(self.preorder(node))
        return res

    # N叉树通用递归模板
    def preorder(self, root: 'Node') -> List[int]:
        res = []

        def helper(root):
            if not root:
                return
            res.append(root.val)
            for child in root.children:
                helper(child)

        helper(root)
        return res


# N叉树迭代方法
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if not root:
            return []
        s = [root]
        # s.append(root)
        res = []
        while s:
            node = s.pop()
            res.append(node.val)
            # for child in node.children[::-1]:
            #     s.append(child)
            s.extend(node.children[::-1])
        return res


def buildTreeByArray(array: List[int]) -> Optional[TreeNode]:
    if len(array) == 0:
        return None

    nodeQueue = []
    root = TreeNode(array[0])
    nodeQueue.append(root)
    lineNodeNum = 2
    startIndex = 1
    restLength = len(array) - 1

    while restLength > 0:
        if restLength < lineNodeNum:
            print("Wrong Input!")
            return None
        i = startIndex
        while i < startIndex + lineNodeNum:
            if i == len(array):
                return root
            cur = nodeQueue.pop(0)
            if array[i] != -1:
                cur.left = TreeNode(array[i])
                nodeQueue.append(cur.left)
            if i + 1 == len(array):
                return root
            if array[i + 1] != -1:
                cur.right = TreeNode(array[i + 1])
                nodeQueue.append(cur.right)
            i = i + 2
            startIndex += lineNodeNum
            restLength -= lineNodeNum
            lineNodeNum = (len(nodeQueue)) * 2
    return root


def mirrorTree(root: TreeNode) -> TreeNode:  # 辅助栈
    if not root:
        return root
    stack = [root]
    while stack:
        node = stack.pop()
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
        node.left, node.right = node.right, node.left
    return root


def mirrorTree2(root: TreeNode) -> TreeNode:  # 递归
    if not root:
        return root
    root.right, root.left = root.left, root.right
    mirrorTree2(root.left)
    mirrorTree2(root.right)
    return root


def isSubPath(self, head: ListNode, root: TreeNode) -> bool:  # 回溯是否存在子路径
    if not root:
        return False

    def back_track(head, root):
        if not head:
            return True
        if not root:
            return False
        if head.val == root.val:
            l = back_track(head.next, root.left)
            if l:
                return True
            r = back_track(head.next, root.right)
            return l | r
        else:
            return False

    return back_track(head, root) | self.isSubPath(head, root.left) | self.isSubPath(head, root.right)


# 判断二叉树是否平衡
def isBalanced(root: TreeNode) -> bool:
    # 计算以当前节点为根的子树树深度/高度
    def Depth(root: TreeNode) -> int:
        if root:
            return 1 + max(Depth(root.left), Depth(root.right))
        return 0
    if not root:
        return True
    return abs(Depth(root.left) - Depth(root.right)) <= 1 and isBalanced(root.left) and isBalanced(root.right)


def leafSimilar(root1: TreeNode, root2: TreeNode) -> bool:
    leaves1 = []
    leaves2 = []

    def leaf(root, nums):
        if root and not root.right and not root.left:
            nums.append(root.val)
        if root.left:
            leaf(root.left, nums)
        if root.right:
            leaf(root.right, nums)
        return nums

    return leaf(root1, leaves1) == leaf(root2, leaves2)


def isSymmetric(root):
    if not root:
        return True

    def dfs(node1: TreeNode, node2: TreeNode) -> bool:
        if not (node1 or node2):
            return False
        if not (node1 and node2):
            return True
        if node1.val != node2.val:
            return False
        return dfs(node1.left, node2.right) and dfs(node1.right, node2.left)

    return dfs(root.left, root.right)


def binaryTreePaths(root):
    """
    :type root: TreeNode
    :rtype: List[str]
    """

    def construct_paths(root, path):
        if root:
            path += str(root.val)
            if not root.left and not root.right:  # 当前节点是叶子节点
                paths.append(path)  # 把路径加入到答案中
            else:
                path += '->'  # 当前节点不是叶子节点，继续递归遍历
                construct_paths(root.left, path)
                construct_paths(root.right, path)

    paths = []
    construct_paths(root, '')
    return paths


def sumNumbers(root):
    """
    :type root: TreeNode
    :rtype: int
    """

    def helper(root):
        if not root:
            return []
        if not root.left and not root.right:
            return [[root.val]]
        lpath = [[root.val] + path for path in helper(root.left)]
        rpath = [[root.val] + path for path in helper(root.right)]
        return lpath + rpath

    paths = helper(root)
    ans = 0
    for path in paths:
        temp = 0
        for val in path:
            temp = temp * 10 + val
        ans += temp
    return ans


if __name__ == "__main__":
    print("This is a Binary Tree test program\n")
    # nums = [1, 2, -1, 3, 4]
    # r = buildTreeByArray(nums)
    # res1 = levelOrder(r)
    # print(res1)
    # mirror_r = mirrorTree2(r)
    # res2 = levelOrder(mirror_r)
    # print(res2)
    # nums = [1, 2, 3]
    # head = buildTreeByArray(nums)
    # res = sumNumbers(head)
    # print(res)
    # res = levelOrder(head)
    # print(res)
    # depth = Depth(head)
    # print(depth)
    # print(isBalanced(head))
    # print(binaryTreePaths(head))
