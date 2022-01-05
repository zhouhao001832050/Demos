from typing import *


# class Solution:
def subarraySum(nums: List[int], k: int) -> int:
    total, ans = 0, 0
    d = {0:1}
    for num in nums:
        total += num
        minusor = total - k
        same = d.get(minusor, 0)
        ans += same
        d[minusor]  = same + 1
    return ans


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = sumi = 0
        d = {0:1}
        for i in range(n):
            sumi += nums[i]
            sumj = sumi - k # 找另一半
            if sumj in d: ans += d[sumj]
            d[sumi] = d.get(sumi,0)+1 #更新dict
        return ans


if __name__ == "__main__":
    nums = [1,2,3,4,5]
    k = 3
    print(subarraySum(nums, k))