from typing import List
from collections import defaultdict
import collections

'''
01背包问题
二分查找
快速排序
两数之和（数组无序）
两数之和（数组有序）
....
'''


class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None


def removeNthFromEnd(head, n):  # 删除链表倒数第n个结点
    global i
    if head is None:
        i = 0
        return None
    head.next = removeNthFromEnd(head.next, n)
    i += 1
    return head.next if i == n else head


def binary_search(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1
    while left <= right:
        pivot = left + (right - left) // 2
        if nums[pivot] == target:
            return pivot
        if target < nums[pivot]:
            right = pivot - 1
        else:
            left = pivot + 1
    return -1


def knapsack(W: List[int], V: List[int]) -> int:  # 01背包
    # 输入样例：
    # W = [4, 1, 2, 3, 4]#重量/体积
    # V = [5, 2, 4, 4, 5]#价值
    n = W[0]  # 物品个数
    m = V[0]  # 背包容量
    dp = [[0 for p in range(m + 1)] for q in range(n + 1)]
    for i in range(1, n + 1):  # 1-n
        for j in range(1, m + 1):  # 0-m
            dp[i][j] = dp[i - 1][j]
            if j >= W[i]:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - W[i]] + V[i])
    print(dp[-1][-1])
    # print(dp[n][m])
    return dp[n][m]


def maxArea(height: List[int]) -> int:
    l, r = 0, len(height) - 1
    ans = 0
    while l < r:
        area = min(height[l], height[r]) * (r - l)
        ans = max(ans, area)
        if height[l] <= height[r]:
            l += 1
        else:
            r -= 1
    return ans


def threeSum(nums: List[int]) -> List[List[int]]:  # 排序+双指针
    n = len(nums)
    nums.sort()  # 需要先排序
    ans = []

    # 枚举 a
    for first in range(n):
        # 需要和上一次枚举的数不相同
        if first > 0 and nums[first] == nums[first - 1]:  # 消除相同的数的效率的影响
            continue
        # c 对应的指针初始指向数组的最右端
        third = n - 1
        target = -nums[first]
        # 枚举 b
        for second in range(first + 1, n):
            # 需要和上一次枚举的数不相同
            if second > first + 1 and nums[second] == nums[second - 1]:
                continue
            # 需要保证 b 的指针在 c 的指针的左侧
            while second < third and nums[second] + nums[third] > target:
                third -= 1
            # 如果指针重合，随着 b 后续的增加
            # 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
            if second == third:
                break
            if nums[second] + nums[third] == target:
                ans.append([nums[first], nums[second], nums[third]])

    return ans


def quick_sort(nums):
    n = len(nums)

    # 快速排序
    def quick(left, right):
        if left >= right:
            return nums
        pivot = left
        i = left
        j = right
        while i < j:
            while i < j and nums[j] > nums[pivot]:
                j -= 1
            while i < j and nums[i] <= nums[pivot]:
                i += 1
            nums[i], nums[j] = nums[j], nums[i]
        nums[pivot], nums[j] = nums[j], nums[pivot]
        quick(left, j - 1)
        quick(j + 1, right)
        return nums

    return quick(0, n - 1)


def two_sum(nums, target):  # 在数组中寻找和为target的两个数并返回它们的数组下标
    # a = [1, 2, 3, 4]
    # print(two_sum(a, 5))
    hash_map = {}  # 构造字典
    result = []  # 若存在多个组合则返回所有结果
    for i, num in enumerate(nums):
        if target - num in hash_map:
            result.append([i, hash_map[target - num]])
        hash_map[num] = i  # 这句不能放在if语句之前，解决list中有重复值或target-num=num的情况
    return result, hash_map


def longestCommonPrefix(strs):
    if not strs:
        return ""
    s1 = min(strs)
    s2 = max(strs)
    for idx, x in enumerate(s1):
        if x != s2[idx]:
            return s2[:idx]
    return s1


def singleNumber(nums: List[int]) -> int:  # 只有一个数字仅出现一次，其余的数字均出现两次
    # 1.交换律：a ^ b ^ c <= > a ^ c ^ b
    #
    # 2.任何数于0异或为任何数
    # 0 ^ n = > n
    #
    # 3.相同的数异或为0: n ^ n = > 0
    #
    # var
    # a = [2, 3, 2, 4, 4]
    #
    # 2 ^ 3 ^ 2 ^ 4 ^ 4
    # 等价于
    # 2 ^ 2 ^ 4 ^ 4 ^ 3 = > 0 ^ 0 ^ 3 = > 3
    a = 0
    for num in nums:
        a = a ^ num
    return a


def singleNumber2(nums: List[int]) -> int:  # 只有一个数字仅出现一次，其余的数字均出现三次
    seen_once = seen_twice = 0

    for num in nums:
        # first appearance:
        # add num to seen_once
        # don't add to seen_twice because of presence in seen_once

        # second appearance:
        # remove num from seen_once
        # add num to seen_twice

        # third appearance:
        # don't add to seen_once because of presence in seen_twice
        # remove num from seen_twice
        seen_once = ~seen_twice & (seen_once ^ num)
        seen_twice = ~seen_once & (seen_twice ^ num)

    return seen_once


def numIdenticalPairs(nums: List[int]) -> int:  # 好数对的数目
    ret, dic = 0, defaultdict(int)
    for i in nums:
        ret, dic[i] = ret + dic[i], dic[i] + 1
        print(ret)
    return ret


def intersect(nums1, nums2):  # 2个数组的交集
    """
    :type nums1: List[int]
    :type nums2: List[int]
    :rtype: List[int]
    """
    if len(nums1) > len(nums2):
        return self.intersect(nums2, nums1)  # 对较短的数组进行元素字典统计
    m = collections.Counter()
    for num in nums1:
        m[num] += 1
    res = []
    for num in nums2:
        if m[num] != 0:
            m[num] = m[num] - 1
            res.append(num)
        else:
            continue
    return res


def maxSubArray1(nums):  # 动态规划法
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    if n == 0:
        return 0
    dp = nums[:]  # ***初始化dp数组，dp[i]存储以nums[i]为结尾的子数组的和的最大值***
    for i in range(1, n):
        dp[i] = max(dp[i - 1] + nums[i], nums[i])

    return max(dp)


def numOfSubarrays(arr, k, threshold):
    """
    :type arr: List[int]
    :type k: int
    :type threshold: int
    :rtype: int
    """
    count = 0
    if len(arr) < k:
        return 0
    add_num = sum(arr[:k])
    target_sum = k * threshold
    if add_num >= target_sum:
        count += 1
    for i in range(k, len(arr)):
        add_num = add_num + arr[i] - arr[i - k]
        if add_num >= target_sum:
            count += 1
    print(count)
    return count


def buildLinkedListByArray(nums: List[int]):
    head = None
    if len(nums) <= 0:
        return head
    head = ListNode(nums[0])
    s = head
    for i in range(1, len(nums)):
        head.next = ListNode(nums[i])
        head = head.next
    head = s
    return head


def reverseLinkedList(head: ListNode) -> ListNode:  # 递归实现
    if not head or not head.next:
        return head
    last = reverseLinkedList(head.next)
    head.next.next = head
    head.next = None
    return last


def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    stack = list()
    edges = defaultdict(list)
    # 0表示未曾访问，1表示正在访问，2表示访问完成
    visit_stat = [0] * numCourses
    has_ring = False
    for e in prerequisites:
        edges[e[1]].append(e[0])

    def dfs(u: int):
        nonlocal has_ring
        visit_stat[u] = 1
        for v in edges[u]:
            if visit_stat[v] == 0:
                dfs(v)
                if has_ring:
                    return
            elif visit_stat[v] == 1:
                has_ring = True
                return
        visit_stat[u] = 2
        stack.append(u)

    for i in range(numCourses):
        if not has_ring and not visit_stat[i]:
            dfs(i)
    if has_ring:
        return []
    else:
        return stack[::-1]


def permute(nums):
    res = []
    track = []

    def backtrack(nums, track):
        if len(track) == len(nums):
            res.append(track.copy())  # 深拷贝
            return
        for i in range(len(nums)):
            if nums[i] in track:
                continue
            track.append(nums[i])
            backtrack(nums, track)
            track.pop()

    backtrack(nums, track)
    return res


def canCompleteCircuit(gas: List[int], cost: List[int]) -> int:
    for i in range(len(gas)):
        if gas[i] < cost[i]:
            continue
        res_gas = gas[i] - cost[i]
        next_i = (i + 1) % len(gas)
        while res_gas >= 0 and next_i != i:
            res_gas += gas[next_i] - cost[next_i]
            next_i = (next_i + 1) % len(gas)
        if res_gas >= 0 and next_i == i:
            return i
        else:
            continue
    return -1


def insertionSortList(head: ListNode) -> ListNode:
    if not head or not head.next:
        return head
    sq = head  # 已经排序的链表
    usq = sq.next  # 未排序的链表
    sq.next = None
    q = usq
    while q:  # 遍历未排序的链表
        pre = ListNode(-1)
        p = sq
        pre.next = p
        while p:  # 寻找已经排序的链表的插入位置
            if q.val <= p.val:
                usq = usq.next
                q.next = p
                pre.next = q
                if p == sq:
                    sq = q
                break
            if not p.next:
                usq = usq.next
                q.next = p.next
                p.next = q
                break
            pre = pre.next
            p = p.next
        q = usq
    return sq


if __name__ == "__main__":
    # W = [4, 1, 2, 3, 4]  # 体积
    # V = [5, 2, 4, 4, 5]  # 价值
    # knapsack(W, V)
    # a = [1, 2, 3, 4, 5, 6, 4]
    # t, di = two_sum(a, 8)
    # arr = [1, 2, 3]
    # head = buildLinkedListByArray(arr)
    # res = findOrder(2, [[1, 0]])
    # print(res)
    # res = canCompleteCircuit([3, 3, 4], [3, 4, 4])
    # print(res)
    # head = buildLinkedListByArray([-1, 5, 3, 4, 0])
    # res = insertionSortList(head)
    # while res:
    #     print(res.val)
    #     res = res.next
    def mycmp(x: int) -> (int, int):
        return rank[x] if x in rank else x


    arr2 = [2, 1, 4, 3, 9, 6]
    n = len(arr2)
    rank = {x: i - n for i, x in enumerate(arr2)}
    print(rank)
    arr1 = [2, 3, 1, 3, 2, 4, 6, 7, 9, 2, 19]
    arr1.sort(key=mycmp)
    print(arr1)
