class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        inf = int(1e9)
        minPrice = inf
        maxProfit = 0
        for price in prices:
            maxProfit = max(price-minPrice, maxProfit)
            minPrice = min(minPrice, price)
        return maxProfit