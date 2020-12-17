from typing import List, Optional
from collections import defaultdict
from collections import Counter


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def buildListByArray(arr):  # 头插法（数组逆序插入）
    head = ListNode(arr[-1])
    for num in arr[:-1]:
        insert_node = ListNode(num)
        insert_node.next = head.next
        head.next = insert_node
    return head


def reverseList(head: ListNode) -> ListNode:  # 递归解法
    if not head or not head.next:
        return head
    p = reverseList(head.next)
    head.next.next = head
    head.next = None
    return p


def reorderList(head):
    """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
    nodes = []
    p = head
    while p is not None:
        nodes.append(p)
        p = p.next
    q = ListNode(-1)
    while nodes:
        first = nodes.pop(0)
        if nodes:
            last = nodes.pop()
        else:
            q.next = first
            first.next = None
            break
        q.next = first
        first.next = last
        q = last
        q.next = None
    return head


def partitionLabels(S):
    end = defaultdict(int)
    for inx, c in enumerate(S):
        end[c] = max(end[c], inx)

    print(end)
    box = []
    i = 0
    while i < len(S):
        start = i
        stop = end[S[i]]
        c = start
        while c < stop:
            if end[S[c]] > stop:
                stop = end[S[c]]
            c += 1
        box.append(stop + 1 - start)
        i = stop + 1
    return box


def isPalindrome(head: ListNode) -> bool:
    s = head
    box = []
    while s:
        box.append(s.val)
        s = s.next
    if not box:
        return True
    length = len(box)
    for i in range(length // 2):
        if box[i] == box[length - 1 - i]:
            continue
        else:
            return False


def missingNumber(nums: List[int]) -> int:
    l = 0
    r = len(nums) - 1
    while l <= r:
        m = (r + l) // 2
        if nums[m] == m:
            l = m + 1
        else:
            r = m - 1
    return l


def videoStitching(clips, T):
    dp = [0] + [float("inf")] * T
    for i in range(1, T + 1):
        for aj, bj in clips:
            if aj < i <= bj:
                dp[i] = min(dp[i], dp[aj] + 1)

    return -1 if dp[T] == float("inf") else dp[T]


def longestMountain(A):
    """
        :type A: List[int]
        :rtype: int
        """
    ans = 0
    flag = [0] * len(A)
    for i in range(1, len(A) - 1):
        if A[i - 1] < A[i] and A[i] > A[i + 1]:
            flag[i] = 1
    for j in range(len(flag)):
        if flag[j] > 0:
            first = j - 1
            last = j + 1
            while first >= 0 and last <= len(flag) - 1:
                if first == 0:
                    if last == len(flag) - 1:
                        break
                    if last < len(flag) - 1:
                        if A[last] > A[last + 1]:
                            last = last + 1
                            continue
                        else:
                            break
                else:
                    if A[first] > A[first - 1]:
                        first = first - 1
                        continue
                    if last < len(flag) - 1:
                        if A[last] > A[last + 1]:
                            last = last + 1
                            continue
                        else:
                            break
                    else:
                        break
            ans = max(ans, last - first + 1)
    return ans


def validMountainArray(A):
    """
    :type A: List[int]
    :rtype: bool
    """
    top = 0
    for i in range(len(A)):
        if i == len(A) - 1:
            return False
        elif A[i] == A[i + 1]:
            return False
        elif A[i] < A[i + 1]:
            top += 1
        else:
            break
    if top < 1:
        return False
    for i in range(top, len(A)):
        if i == len(A) - 1:
            return True
        elif A[i] == A[i + 1]:
            return False
        elif A[i] > A[i + 1]:
            continue
        else:
            return False


if __name__ == "__main__":
    head = buildListByArray([1])
    # reorderList(head)
    # S = "qiejxqfnqceocmy"
    # res = partitionLabels(S)
    # print(isPalindrome(head))
    # nums = [1]
    # print(missingNumber(nums))
    # clips = [[0, 2], [4, 6], [8, 10], [1, 9], [1, 5], [5, 9]]
    # T = 10
    # res = videoStitching(clips, T)
    # a = [1, 2, 1]
    # res = longestMountain(a)
    # print(res)
    test = [3, 6, 9, 1]
    t = sorted(test)
    print(t)
