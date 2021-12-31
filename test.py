import os

def try_iterate(s):
    seq_length = len(s)
    for idx in range(seq_length-1, -1, -1):
        print(s[idx])


if __name__ == '__main__':
    s = "abcdefg"
    try_iterate(s)
    