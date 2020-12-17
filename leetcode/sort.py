# 几种经典的排序算法
# 需要重点关注快速排序、归并排序
# 其次是桶排序、基数排序、希尔排序
import collections
from typing import List
import heapq
import math


def selection_sort(nums):
    # 选择排序
    n = len(nums)
    for i in range(n):
        for j in range(i, n):
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
    return nums


def bubble_sort(nums):
    # 冒泡排序
    n = len(nums)
    # 进行多次循环
    for c in range(n):
        for i in range(1, n - c):
            if nums[i - 1] > nums[i]:
                nums[i - 1], nums[i] = nums[i], nums[i - 1]
    return nums


def insertion_sort(nums):
    # 插入排序
    n = len(nums)
    for i in range(1, n):
        while i > 0 and nums[i - 1] > nums[i]:
            nums[i - 1], nums[i] = nums[i], nums[i - 1]
            i -= 1
    return nums


def shell_sort(nums):
    # 希尔排序
    n = len(nums)
    gap = n // 2
    while gap:
        for i in range(gap, n):
            while i - gap >= 0 and nums[i - gap] > nums[i]:
                nums[i - gap], nums[i] = nums[i], nums[i - gap]
                i -= gap
        gap //= 2
    return nums


def merge_sort(nums):
    def merge(left, right):  # 合并2个有序数组
        res = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1
        res += left[i:]
        res += right[j:]
        return res

    # 归并排序
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    # 分
    left = merge_sort(nums[:mid])  # 2路归并排序
    right = merge_sort(nums[mid:])
    # 合并
    return merge(left, right)


def quick_sort(nums):
    n = len(nums)

    # 快速排序
    def quick(left, right):
        if left >= right:  # 1
            return nums
        pivot = left  # 2
        i = left
        j = right
        while i < j:  #
            while i < j and nums[j] >= nums[pivot]:  # 3
                j -= 1
            while i < j and nums[i] <= nums[pivot]:  # 4
                i += 1
            nums[i], nums[j] = nums[j], nums[i]  # 5
        nums[pivot], nums[j] = nums[j], nums[pivot]  # 6
        quick(left, j - 1)
        quick(j + 1, right)
        return nums

    return quick(0, n - 1)


def heap_sort(nums):
    # 堆排序
    # 调整堆
    # 迭代写法
    # def adjust_heap(nums, startpos, endpos):
    #     newitem = nums[startpos]
    #     pos = startpos
    #     childpos = pos * 2 + 1
    #     while childpos < endpos:
    #         rightpos = childpos + 1
    #         if rightpos < endpos and nums[rightpos] >= nums[childpos]:
    #             childpos = rightpos
    #         if newitem < nums[childpos]:
    #             nums[pos] = nums[childpos]
    #             pos = childpos
    #             childpos = pos * 2 + 1
    #         else:
    #             break
    #     nums[pos] = newitem

    # 递归写法
    def adjust_heap(nums, startpos, endpos):
        pos = startpos
        chilidpos = pos * 2 + 1
        if chilidpos < endpos:
            rightpos = chilidpos + 1
            if rightpos < endpos and nums[rightpos] > nums[chilidpos]:
                chilidpos = rightpos
            if nums[chilidpos] > nums[pos]:
                nums[pos], nums[chilidpos] = nums[chilidpos], nums[pos]
                adjust_heap(nums, pos, endpos)

    n = len(nums)
    # 建堆
    for i in reversed(range(n // 2)):
        adjust_heap(nums, i, n)
    # 调整堆
    for i in range(n - 1, -1, -1):
        nums[0], nums[i] = nums[i], nums[0]
        adjust_heap(nums, 0, i)
    return nums


def counting_sort(nums):
    # 计数排序
    # 最大值和与最小值之差过大且数组数目较小时不适用
    # 空间利用率底
    if not nums:
        return []
    n = len(nums)
    _min = min(nums)
    _max = max(nums)
    tmp_arr = [0] * (_max - _min + 1)
    for num in nums:
        tmp_arr[num - _min] += 1
    j = 0
    for i in range(n):
        while tmp_arr[j] == 0:
            j += 1
        nums[i] = j + _min
        tmp_arr[j] -= 1
    return nums


def bucket_sort(nums, bucketSize):
    # 桶排序
    if len(nums) < 2:
        return nums
    _min = min(nums)
    _max = max(nums)
    # 需要桶个数
    bucketNum = (_max - _min) // bucketSize + 1
    buckets = [[] for _ in range(bucketNum)]
    for num in nums:
        # 放入相应的桶中
        buckets[(num - _min) // bucketSize].append(num)
    res = []

    for bucket in buckets:
        if not bucket:
            continue
        if bucketSize == 1:
            res.extend(bucket)
        else:
            # 当都装在一个桶里,说明桶容量大了
            if bucketNum == 1:
                bucketSize -= 1
            res.extend(bucket_sort(bucket, bucketSize))
    return res


def Radix_sort(nums):
    # 基数排序
    if not nums:
        return []
    _max = max(nums)
    # 最大位数
    maxDigit = len(str(_max))
    bucketList = [[] for _ in range(10)]
    # 从低位开始排序
    div, mod = 1, 10
    for i in range(maxDigit):
        for num in nums:
            bucketList[num % mod // div].append(num)
        div *= 10
        mod *= 10
        idx = 0
        for j in range(10):
            for item in bucketList[j]:
                nums[idx] = item
                idx += 1
            bucketList[j] = []
    return nums


def subSort(array: List[int]) -> List[int]:
    if len(array) <= 1:
        return [-1, -1]
    # 从左往右找右端点
    ma = array[0]
    r = -1
    n = len(array)
    for i in range(n):
        if array[i] >= ma:
            ma = array[i]
        else:
            r = i
    if r == -1:
        return [-1, -1]
    mi = array[n - 1]
    l = -1
    for i in range(n - 1, -1, -1):
        if array[i] <= mi:
            mi = array[i]
        else:
            l = i
    return [l, r]


def quick_sort_test(nums):
    def qs(l, r):
        if l >= r:  # 递归结束条件
            return
        pivot = l  # 选择数组第一个元素用于划分
        i = l
        j = r
        while i < j:
            # nums[j] >= nums[pivot]中的‘等于’很重要，可以防止数组中存在重复元素时的死循环
            # 为什么一定要从右边开始？
            while i < j and nums[j] >= nums[pivot]:  # 从右边开始，找到第一个比nums[pivot]小的元素
                j -= 1
            while i < j and nums[i] <= nums[pivot]:  # 随后从左边开始，找到第一个比nums[pivot]大的元素
                i += 1
            nums[i], nums[j] = nums[j], nums[i]  # 交换两边各自找到的元素

        nums[pivot], nums[j] = nums[j], nums[pivot]  # 当所有不满足当前条件的元素都交换后，将支点元素填入j（？为什么是j）
        qs(l, i - 1)  # 左半区递归
        qs(i + 1, r)  # 右半区递归
        return

    n = len(nums)
    qs(0, n - 1)
    return nums


def reorganizeString(S: str) -> str:
    if len(S) < 2:
        return S

    length = len(S)
    counter = collections.Counter(S)
    maxCount = max(counter.items(), key=lambda x: x[1])[1]
    if maxCount > (length + 1) // 2:
        return ""

    queue = [(-x[1], x[0]) for x in counter.items()]
    heapq.heapify(queue)  # 大根堆
    ans = list()

    while len(queue) > 1:
        _, letter1 = heapq.heappop(queue)
        _, letter2 = heapq.heappop(queue)
        ans.extend([letter1, letter2])
        counter[letter1] -= 1
        counter[letter2] -= 1
        if counter[letter1] > 0:  # 若该字母的数量依然大于零，则重新加入堆
            heapq.heappush(queue, (-counter[letter1], letter1))
        if counter[letter2] > 0:
            heapq.heappush(queue, (-counter[letter2], letter2))

    if queue:
        ans.append(queue[0][1])

    return "".join(ans)


def countPrimes(n: int) -> int:
    # def isPrime(a: int):
    #     for i in range(2, a):
    #         if a % i == 0:
    #             return False
    #     return True
    #
    # for num in range(2, n):
    #     if isPrime(num):
    #         print(num)
    #         ans += 1
    # return ans
    if n < 2:
        return 0
    nums = [1] * n
    # 0和1不是质数(素数)
    nums[0] = nums[1] = 0
    for i in range(2, int(math.sqrt(n) + 1)):
        if nums[i]:  # 如果i是质数
            # 因为i是质数，所以我们需要把2*i, 3*i, 4*i等这些i的倍数给删除掉
            # 比如i=2时，我们要删除4, 6, 8等
            # 当i=3时，我们要删除6, 9, 12等，发现了吧，6重复了删除了，
            # 所以不管i是什么，我们从i*i开始就可以避免重复，i=3时就从i*i=9开始删除
            #
            for j in range(i * i, n, i):
                nums[j] = 0
            # nums[i * i:n:i] = [0] * ((n - 1 - i * i) // i + 1)
            # 上面是python的写法，啰嗦一下，就是将i*i到n-1之间步长为i的所有数给删除掉
            # 因为i是质数，所以i*i，i*i+i，i*i+2i, i*i+3i等这些小于n的步长为i的数都不是质数
            # 它们就等价于k*i, (k+1)*i, (k+2)*i等等i的倍数
    return sum(nums)


def isPossible(nums: List[int]) -> bool:
    # 哈希表 + 堆
    hash_map = collections.defaultdict(list)
    for num in nums:
        if hash_map.get(num - 1, []):
            prev_min_len = heapq.heappop(hash_map[num - 1])
            heapq.heappush(hash_map[num], prev_min_len + 1)
        else:
            heapq.heappush(hash_map[num], 1)

    for num in hash_map:
        if hash_map[num] and heapq.heappop(hash_map[num]) < 3:
            return False

    return True


def matrixScore(A: List[List[int]]) -> int:
    c = len(A)
    r = len(A[0])
    count = [0 for _ in range(r)]
    ans = 0
    for i in range(c):
        if A[i][0] == 0:
            for j in range(r):
                if A[i][j] == 1:
                    A[i][j] = 0
                else:
                    A[i][j] = 1
        for k in range(r):
            if A[i][k] == 1:
                count[k] += 1
    for num in range(len(count)):
        if count[num] > c // 2:
            ans += (2 ** (r - 1 - num)) * count[num]
        else:
            ans += (2 ** (r - 1 - num)) * (c - count[num])
    return ans


def splitIntoFibonacci(S: str) -> List[int]:
    ans = list()

    def backtrack(index: int):
        if index == len(S):  # 回溯算法本质为递归，因此必须有递归出口
            return len(ans) >= 3

        curr = 0  # 用于累加，标识当前字符串的数值
        for i in range(index, len(S)):
            if i > index and S[index] == "0":  # 防止以0开头的数字出现
                break
            curr = curr * 10 + ord(S[i]) - ord("0")
            if curr > 2 ** 31 - 1:  # 单个数值上限
                break
            if len(ans) < 2 or curr == ans[-2] + ans[-1]:  # 什么情况下选择
                ans.append(curr)  # 做选择
                if backtrack(i + 1):  # 子递归满足条件即可返回
                    return True
                ans.pop()  # 撤销选择
            elif len(ans) >= 2 and curr > ans[-2] + ans[-1]:  # 对于curr > ans[-2] + ans[-1] 若小于，还可能通过增加位数使之相等，大于则不可能
                break
        return False

    backtrack(0)
    return ans


def monotoneIncreasingDigits(N: int) -> int:
    strN = [c for c in str(N)]
    i = 1  # 为什么是1不是0
    while i < len(strN) and strN[i - 1] <= strN[i]:
        i += 1
    if i < len(strN):  # 为什么会有这一判断
        while i > 0 and strN[i - 1] > strN[i]:
            strN[i - 1] = str(int(strN[i - 1]) - 1)
            i -= 1
        for j in range(i + 1, len(strN)):
            strN[j] = '9'
    return int("".join(strN))


def maxProfit(prices: List[int], fee: int) -> int:
    n = len(prices)
    # 滚动数组优化空间复杂度为O(1)
    # dp0 = 0
    # dp1 = -prices[0]
    # for i in range(1, n):
    #     dp0 = max(dp0, dp1 + prices[i] - fee)
    #     dp1 = max(dp1, dp0 - prices[i])
    # 标准dp
    dp = [[0, -prices[0]]] + [[0, 0] for _ in range(n - 1)]
    for i in range(1, n):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee)
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[n - 1][0]


if __name__ == "__main__":
    arr = [1, 2, 4, 7, 10, 12, 7, 12, 6, 7, 12, 18, 19]
    # res = subSort(arr)
    prices = [1, 3, 2, 8, 4, 9]
    fee = 2
    val = maxProfit(prices, fee)
    print(val)
