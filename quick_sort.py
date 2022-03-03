
# # partition
# def partition(left, right, nums):
#     pivot = nums[left]
#     i = left
#     j = right
#     while i < j:
#         while i<j and nums[j]>=pivot:
#             j -= 1
#         nums[i] = nums[j]
#         while i < j and nums[i]<pivot:
#             i += 1
#         nums[j] = nums[i]
#     nums[i] = pivot

#     return i

# def quick_sort(nums, left, right):
#     if left< right:
#         index = partition(left, right, nums)

#         quick_sort(nums, index+1, right)
#         quick_sort(nums, left, index-1)
#     return nums


# if __name__ == "__main__":
#     nums = [8,3,2,1,5,6,7,8,9]
#     right = len(nums)-1
#     print(quick_sort(nums, 0, right))

def get_sqrt(n, e):
    """
    :param n:
    :param e:
    :return:
    """
    min, max = 0, n
    mid = (min+max)/2
    while abs(mid*mid - n) > e:
        squares = mid * mid
        if squares > n:
            max = mid
        elif squares < n:
            min = mid
        else:
            return mid
        mid = (min+max) / 2
    return mid


if __name__ == "__main__":
    n = 3
    e = 1e-7
    print(get_sqrt(n, e))
