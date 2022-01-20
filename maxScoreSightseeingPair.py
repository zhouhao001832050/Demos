import os

def maxScoreSightseeingPair(self, A)
    res = 0
    preMax = A[0] + 0
    for j in range(1, len(A)):
        res = max(res, preMax + A[j]-j)
        preMax = max(preMax, A[j]+j)
    return res