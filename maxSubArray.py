from typing import List

def maxSubArray(nums: List[int]) -> int:
    dp = [0 for _ in range(nums)]
    dp[0] = nums[0]
    size = len(nums)
    if size == 0:
        return 0
    for i in range(1, size):
        if dp[i-1] > 0:
            dp[i] = dp[i-1] + nums[i]
        else:
            dp[i] = nums[i]
    return max(dp)


def maxSubArray(nums: List[int]) -> int:
    pre = 0
    size = len(nums)
    res = nums[0]
    for i in range(size):
        pre = max(nums[i], pre + nums[i])
        res = max(pre, res)
    return res