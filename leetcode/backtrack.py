def permute(nums):
    res = []

    def backtrack(path, nums):
        if len(path) == len(nums):
            res.append(path.copy())  #
            return
        for num in nums:
            if num in path:
                continue
            path.append(num)
            backtrack(path, nums)
            path.pop()

    backtrack([], nums)
    return res


def fib(n):
    if n == 1 or n == 2:
        return 1
    return fib(n - 2) + fib(n - 1)


def dp_fib(n):  # 自顶向下
    if n < 1:
        return 0
    dp = [0] * (n + 1)

    def helper(meno, n):
        if n == 1 or n == 2:
            return 1
        if meno[n] != 0:
            return meno[n]
        meno[n] = helper(meno, n - 1) + helper(meno, n - 2)
        return meno[n]

    helper(dp, n)
    return dp[n]


def fib3(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


if __name__ == "__main__":
    arr = [1, 2, 3]
    print(permute(arr))
