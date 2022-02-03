
def minDistance(word1, word2):
    m = len(word1)
    n = len(word2)

    # 字符串为空
    if m*n == 0:
        return m+n
    
    # DP数组
    dp = [ [0] * (m+1) for _ in range(n+1) ]

    # 边界状态初始化
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    
    # 计算所有dp值
    for i in range(1, n+1):
        for j in range(1, m+1):
            left = dp[i-1][j] + 1
            down = dp[i][j-1] + 1
            left_down = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                left_down += 1
            dp[i][j] = min(left, down, left_down)
    return dp[n][m]