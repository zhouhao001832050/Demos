import os
from typing import *

# 超时代码
def productExceptMe(nums: List[int]) -> List[int]:
    prefixes = [0] * len(nums)
    sufixes = [0] * len(nums)
    for i, num in enumerate(nums):
        if i == 0:
            prefixes[i] = 1
        else:
            prefixes[i] = prefixes[i-1] * nums[i-1]
    for j, num in enumerate(nums[::-1]):
        if j == 0:
            sufixes[j] = 1
        else:
            sufixes[j] = sufixes[j-1] * nums[::-1][j-1]
    # import pdb;pdb.set_trace()
    sufixes = sufixes[::-1]
    return [sufixes[i] * prefixes[i] for i, num in enumerate(nums)]



class Solution:
    def productExceptMe(self, nums: List[int]) -> List[int]:
        ans = [0] * len(nums)
        ans[0] = 1
        for i in range(1, len(nums)):
            ans[i] = ans[i-1] * nums[i-1]

        Right = 1
        for i in reversed(range(len(ans))):
            ans[i] *= Right
            Right *= nums[i]
        # import pdb;pdb.set_trace()
        return ans


if __name__ == "__main__":
    nums = [1,2,3,4]
    res = productExceptMe(nums)
    print(res)
