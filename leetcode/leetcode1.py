# -*- coding: UTF-8 -*-
# 算法题代码汇总
# 2020.03.06
from LEETCODE.BinaryTree import TreeNode
import copy


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def generate_list(nums):
    head = ListNode(-1)
    for num in nums:
        new_node = ListNode(num)
        s = head.next
        head.next = new_node
        new_node.next = s
    return head


def show_list(head):
    while head:
        print(head.val)
        head = head.next


def subsets(nums, target):  # 求某集合所有满足（sum=target）条件的所有子集
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    # num = [1, 2, 2, 6, 2, 3, 4]
    # print(subsets(num, 6))
    n = len(nums)
    res = []
    list_one = []
    nums.sort(reverse=False)
    backtrack(0, list_one, nums, n, res, target)
    return res


def backtrack(i, list_one, nums, n, res, target):  # 回溯(未实现剪枝)
    if sum(list_one) == target and list_one not in res:
        res.append(list_one[:])  # 此处list_one[:]为列表的深拷贝；若只是采用列表名list_one,则重复将该列表加入results列表中，results中所有元素都一样
    for j in range(i, n):  # 递归结束条件，这里即为j>=n
        list_one.append(nums[j])
        backtrack(j + 1, list_one, nums, n, res, target)
        if len(list_one) is not 0:
            list_one.pop()


def generate(item, n, results):
    # results = []
    # generate('', 2, results)
    # print(results)
    if len(item) == 2 * n:
        results.append(item)
    generate(item + '(', n, results)
    generate(item + ')', n, results)


class Solution(object):

    def maxDepth(self, root):  # 求一课二叉树的最大深度（递归）
        if not root:
            return 0
        else:
            left_height = self.maxDepth(root.left)
            right_height = self.maxDepth(root.right)
            return max(left_height, right_height) + 1

    def middleNode(self, head):  # 返回链表中间结点（快慢指针）
        fast = head
        slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def buildTree(self, preOrder, inOrder):  # 根据中序遍历与前序遍历的结果重新建立二叉树
        """
        :type preOrder: List[int]
        :type inOrder: List[int]
        :rtype: TreeNode
        """
        # s1 = Solution()
        # Btree = s1.buildTree([3, 9, 20, 15, 7], [9, 3, 15, 20, 7])
        if len(preOrder) == 0:
            return None
        # 创建当前节点
        node = TreeNode(0)
        # 在中序遍历中查找第一个节点的位置
        node.val = preOrder[0]
        index = inOrder.index(preOrder[0])
        # 划分左右子树
        left_pre = preOrder[1:index]
        left_in = inOrder[:index]
        right_pre = preOrder[index + 1:]
        right_in = inOrder[index + 1:]
        # 遍历创建子树
        node.left = self.buildTree(left_pre, left_in)
        node.right = self.buildTree(right_pre, right_in)
        # 返回当前节点
        return node
        #      1.前序遍历的顺序：根结点->左子树->右子树；中序遍历的顺序：左子树->根节点->右子树
        #      2.每一次划分操作后左子树（或右子树）的中序遍历列表元素与前序遍历的元素的个数相同
        #      3.每次以前序遍历的第一个元素进行左右子树集合划，由此2个遍历列表中的元素都被划分为2个部分。
        #      4.需要注意的是，前序遍历在划分时是从第二元素（preOrder[1:index]）开始的；之后再根据中序遍历的划分结果（个数特征）来划分为2部分的。

    def exist(self, board: [[str]], word: str) -> bool:
        m, n = len(board), len(board[0])
        visited = [[0] * n for _ in range(m)]

        def dfs(kth, i, j):
            if kth == len(word): return True
            res = False
            if 0 <= i < m and 0 <= j < n and board[i][j] == word[kth] and not visited[i][j]:
                visited[i][j] = 1
                res = dfs(kth + 1, i + 1, j) or dfs(kth + 1, i - 1, j) or dfs(kth + 1, i, j + 1) or dfs(kth + 1, i,
                                                                                                        j - 1)
                if not res: visited[i][j] = 0
            return res

        for i in range(m):
            for j in range(n):
                if dfs(0, i, j):
                    return True
        return False


def subsets2(nums):  # 数学归纳法求（不含重复元素）集合子集
    # a = [1, 2, 3, 4]
    # print(subsets2(a))
    if len(nums) == 0:
        return [[]]
    elem = nums.pop()
    res = subsets2(nums)  # 递归
    size = len(res)
    for k in range(size):
        res.append(res[k][:])  # 注意：在把原先的res列表中的列表对象逐一拷贝的过程中,需要使用深拷贝,否则原先列表列表中保存的数据也会发生改变
        res[-1].append(elem)  # 时间复杂度O(N*2**N)
    return res


def two_sum(nums, target):  # 在数组中寻找和为target的两个数并返回它们的数组下标
    # a = [1, 2, 3, 4]
    # print(two_sum(a, 5))
    hash_map = {}  # 构造字典
    result = []  # 若存在多个组合则返回所有结果
    for i, num in enumerate(nums):
        if target - num in hash_map:
            result.append([i, hash_map[target - num]])
        hash_map[num] = i  # 这句不能放在if语句之前，解决list中有重复值或target-num=num的情况
        print(hash_map)  # 字典实现哈希映射（hash_map）
    return result


def add_two_num(l1, l2):  # 两数之和（链表逐位相加）
    # a = generate_list([2, 4, 3])  # 79
    # b = generate_list([5, 6, 4])  # 99
    # l3 = add_two_num(a, b)  # 178
    # show_list(l3.next)
    new_list = ListNode(-1)
    r = new_list
    p = l1.next
    q = l2.next
    carry = 0
    while p or q:
        if p:
            l1_val = p.val
            p = p.next
        else:
            l1_val = 0
        if q:
            l2_val = q.val
            q = q.next
        else:
            l2_val = 0
        sum_val = l1_val + l2_val + carry
        carry = sum_val // 10
        val = sum_val % 10
        r.next = ListNode(val)
        r = r.next
    if carry is not 0:
        r.next = ListNode(carry)
        r = r.next
    return new_list


def mergeTrees(t1, t2):
    if not t1:
        return t2
    if not t2:
        return t1
    t1.val += t2.val
    t1.left = mergeTrees(t1.left, t2.left)
    t1.right = mergeTrees(t1.right, t2.right)
    return t1 or t2
    # res1 = (1 or 3)  # return 1 (or： 若参与运算的两者一个为0/None,另外一个非0.则返回0.否则返回第一个值)
    # res2 = (1 and 3)  # return 3 (and： 若参与运算地两者一个为0/None,另外一个非0.则返回非0值.否则返回第二个值)
    # res3 = (1 or 0)  # return 1
    # res4 = (0 and 3)  # return 0


def coin_change(coins, amount):
    # print(coin_change([2, 5, 7], 15))
    result = [0]  # 初始条件
    for i in range(1, amount + 1):
        result.append(float('inf'))
        for j in coins:
            if i >= j and result[i - j] != float('inf'):
                result[i] = min(result[i - j] + 1, result[i])
    if result[amount] == float('inf'):
        result[amount] = -1

    return result


def find_repeat_number(nums):
    # print(find_repeat_number([2, 2, 3, 4, 2, 4, 3]))
    result = []
    temp = []
    for num in nums:
        if num in temp:
            result.append(num)
        temp.append(num)
    return result


def lengthOfLongestSubstring(s: str) -> int:
    st = {}
    i, ans = 0, 0
    for j in range(len(s)):
        if s[j] in st:
            i = max(st[s[j]], i)
        ans = max(ans, j - i + 1)
        st[s[j]] = j + 1
    return ans


def twsum(nums, target):
    hash_table = {}
    n = len(nums)
    for i in range(n):
        temp = target - nums[i]
        if temp in hash_table:
            return [i, hash_table[temp]]
        hash_table[nums[i]] = i
    return None


# def findMedianSortedArrays(nums1, nums2):
#     m = len(nums1)
#     n = len(nums2)


def mergeTwoSortedArrays(nums1, nums2):
    result = []
    i = j = 0
    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            result.append(nums1[i])
            i += 1
        else:
            result.append(nums2[j])
            j += 1
    if i >= len(nums1):
        for elem in nums2[j:]:
            result.append(elem)
    if j >= len(nums2):
        for elem in nums1[i:]:
            result.append(elem)
    return result


def findMedianSortedArrays(nums1, nums2):
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: float
    """
    nums = mergeTwoSortedArrays(nums1, nums2)
    if len(nums) % 2 == 0:
        return (nums[int(len(nums) / 2)] + nums[int((len(nums)) / 2 - 1)]) / 2
    if len(nums) % 2 != 0:
        return nums[int(len(nums) / 2)]


def longestPalindrome(s: str) -> str:
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    ans = ""
    # 枚举子串的长度 l+1
    for l in range(n):
        # 枚举子串的起始位置 i，这样可以通过 j=i+l 得到子串的结束位置
        for i in range(n):
            j = i + l
            if j >= len(s):
                break
            if l <= 1:  # 只有一个字符
                if l == 0:
                    dp[i][j] = True
                if l == 1:  # 2个字符的情况
                    dp[i][j] = (s[i] == s[j])
            else:  # 状态转移(需要排除)
                dp[i][j] = (dp[i + 1][j - 1] and s[i] == s[j])
            if dp[i][j] and l + 1 > len(ans):
                ans = s[i:j + 1]
    return ans


def reverse_better(x: int) -> int:  # 整数反转。考虑溢出
    y, res = abs(x), 0
    # 数值范围为 [−2^31,  2^31 − 1]
    boundry = (1 << 31) - 1 if x > 0 else 1 << 31  # 1<<31=2^31
    while y != 0:
        res = res * 10 + y % 10
        if res > boundry:
            return 0
        y //= 10
    return res if x > 0 else -res


def fun(s: str) -> int:  # 实现将一个字符串变为回文串的最少需要改变的字符的个数
    res = 0
    if len(s) == 0:
        print("字符串为空")
        return res
    elif len(s) == 1:
        return res
    else:
        i = 0
        j = len(s) - 1
        while i <= j:
            if s[i] != s[j]:
                res += 1
            i += 1
            j -= 1
    return res


def canMakePaliQueries(s, queries):
    res = []
    for query in queries:
        st = s[query[0]:query[1]]
        print(fun(st))
        if query[2] >= fun(st):
            res.append(True)
        else:
            res.append(False)
    return res


def countNodes(root: TreeNode) -> int:
    if not root:
        return 0
    q = [root]
    ans = 0
    while not q:
        n = len(q)
        ans += n
        for i in range(n):
            node = q.pop(0)
            if not node.left:
                q.append(node.left)
            if not node.right:
                q.append(node.right)
    return ans


def solveNQueens(n):
    board = [["."] * n for _ in range(n)]
    ans = []

    def backtrack(board, row):  # 回溯法
        if row == len(board):  # 满足条件
            temp = []
            for r in board:
                s = "".join(r)
                temp.append(s)
            ans.append(temp)
            return
            # return True
        n = len(board[row])
        for i in range(n):
            if not isValid(board, row, i):
                continue
            board[row][i] = "Q"
            backtrack(board, row + 1)
            # if backtrack(board, row + 1):
            #     return True
            board[row][i] = "."
        # return False

    def isValid(board, row, col):
        n = len(board)
        for i in range(n):
            if board[i][col] == "Q":  # 列冲突
                return False
        # 判断对角线冲突应当只考虑已经尝试过放置皇后的行与列
        # 即row的范围为：[0,row-1]
        #   col的范围为：[0,n-1]

        x1 = row - 1  # 思考：为什么不是+1?
        y1 = col - 1
        while x1 >= 0 and y1 >= 0:  # 主对角线冲突
            if board[x1][y1] == "Q":
                return False
            x1 -= 1
            y1 -= 1

        x2 = row - 1
        y2 = col + 1
        while x2 >= 0 and y2 < n:  # 副对角线冲突
            if board[x2][y2] == "Q":
                return False
            x2 -= 1
            y2 += 1
        return True

    backtrack(board, 0)
    return ans


if __name__ == "__main__":
    # a = generate_list([2, 4, 3])  # 79
    # b = generate_list([5, 6, 4])  # 99
    # l3 = add_two_num(a, b)  # 178
    # show_list(l3.next)
    # str1 = "dvdkdcef"
    # print(lengthOfLongestSubstring(str1))
    # matrix = [[5, 10, 5, 4, 4], [1, 7, 8, 4, 0], [3, 4, 9, 0, 3]]
    # print(matrix[:][1])
    # print(mergeTwoSortedArrays([1, 2], [3, 4]))
    # print(findMedianSortedArrays([1, 2], [3, 4]))
    # print(longestPalindrome("awsds"))
    # s = "abcda"
    # queries = [[3, 3, 0], [1, 2, 0], [0, 3, 1], [0, 3, 2], [0, 4, 1]]
    # res = canMakePaliQueries(s, queries)
    # print(res)
    res = solveNQueens(4)
    print(res)

