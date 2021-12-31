from typing import *

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        votes, count = 0, 0
        for num in nums:
            if votes == 0: x = num
            votes += 1 if num == x else -1
        print(x)
        # 验证 x 是否为众数
        for num in nums:
            if num == x: count += 1
        return x if count > len(nums) // 2 else 0 # 当无众数时返回 0


if __name__ == '__main__':
    s = Solution()
    nums = [3,2,3,4,2,2]
    res = s.majorityElement(nums)
    print(res)