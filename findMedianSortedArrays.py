from typing import *

def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    res = []
    point1, point2 = 0,0

    while point1 < len(nums1) or point2 < len(nums2):
        if point1 == len(nums1):
            res.append(nums2[point2])
            point2 += 1
        elif point2 == len(nums2):
            res.append(nums1[point1])
            point1 += 1

        elif nums1[point1] <= nums2[point2]:
            res.append(nums1[point1])
            point1 += 1
        else:
            res.append(nums2[point2])
            point2 += 1
    
    nums1[:] = res
    print(nums1)
    if len(nums1) % 2 == 1:
        ans = nums1[len(nums1) // 2]
    else:
        ans = (nums1[len(nums1) // 2] + (nums1[len(nums1) // 2 -1])) / 2
    print(ans)
    return ans

if __name__ == "__main__":
    nums1 = [1,2,3,4,5]
    nums2 = [4,5,6,7,8]
    findMedianSortedArrays(nums1, nums2)