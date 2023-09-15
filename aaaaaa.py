# import numpy as np


# preds = np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])


# preds_arg = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
# preds_max = np.max(preds, axis=1)
# print(preds_arg)
# print(preds_max)
from collections import defaultdict
def find_missing_gem(remaining_gems, all_gems):
    dic1 = defaultdict(int)
    dic2 = defaultdict(int)
    for i in remaining_gems:
        dic1[i]+=1
    for j in all_gems:
        dic2[j]+=1    
    for s,num in dic2.items():
        if s not in dic1:
            return s
        elif num !=dic1[s]:
            return s
        else:
            continue

# 输入剩余的宝石和全部的宝石收藏
remaining_gems, all_gems = input().split()
# 
# 查找丢失的宝石字母
missing_gem = find_missing_gem(remaining_gems, all_gems)

# 输出丢失的宝石字母
print(missing_gem)
