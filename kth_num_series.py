import os

# 这个脚本是算法题第k大/小的系列， 将提供出写这一系列问题的解法
def partition(nums,left, right):
    pivot = nums[left]
    i,j=left, right
    while i < j:
        while i<j and nums[j]>=pivot:
            j -= 1
        nums[i] = nums[j]
        while i<j and nums[i] <= pivot:
            i += 1
        nums[j] = nums[i]
    nums[i] = pivot
    return i


def topk_split(nums, k, left, right):
    if left < right:
        index = partition(nums, left, right):
        if index == k:
            return 
        elif index < k:
            topk_split(nums, k, index+1, right)
        else:
            topk_split(nums, k, left, index-1)
    

def kth_largest(nums, k):
    topk_split(nums, len(nums)-k, 0, len(nums))
    return nums[len(nums)-k]



