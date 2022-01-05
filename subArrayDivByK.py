import os



# look for preSum[i] mod k == preSum[j] mod k
def subArrayDivByK(nums, k):
    d = {0:1}
    ans = 0
    total = 0
    for num in nums:
        total += num
        remainder = total % k
        same = d.get(remainder, 0)
        ans += same
        d[remainder] = same + 1
    return ans



