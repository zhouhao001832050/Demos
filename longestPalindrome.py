import os

class Solution:
    def longestPalindrome(self, s: str) -> str:
        size = len(s)
        if size <= 1:
            return s
        # 创建动归 dp
        dp = [[False for _ in range(size)] for _ in range(size)]
        max_len = 1
        for j in range(1, size):
            for i in range(j):
                if j - i <= 2:
                    if s[i] == s[j]:
                        dp[i][j] = True
                        cur_len = j - i + 1
                else:
                    if s[i] == s[j] and dp[i+1][j-1]:
                        dp[i][j] = True
                        cur_len = j - i + 1
                if dp[i][j]:
                    if cur_len > max_len:
                        max_len = cur_len
                        start = i
        return s[start:start+max_len]


if __name__ == "__main__":
    a = Solution()
    s = "aabbaacfghtewabcdefgfedcba"
    print(a.longestPalindrome(s))