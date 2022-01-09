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



def longestPalindrome_2nd(s: str) -> str:
    size = len(s)
    if size == 1:
        return s
    start = 0
    # 创建动态规划dynamic programing表
    dp = [[False for _ in range(size)] for _ in range(size)]
    # 初始长度为1，考虑到如果没有找到就可以返回1
    max_len = 1
    for j in range(1, size):
        for i in range(j):
            # 边界条件：
            # 只要头尾相等（s[i]==s[j]）就能返回True
            if j-i <= 2:
                if s[j] == s[i]:
                    dp[i][j] = True
                    cur_len = j - i + 1
            # 状态转移方程 
            # 当前dp[i][j]状态：头尾相等（s[i]==s[j]）
            # 过去dp[i][j]状态：去掉头尾之后还是一个回文（dp[i+1][j-1] is True）
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