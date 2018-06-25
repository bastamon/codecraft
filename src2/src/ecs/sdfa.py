import os
import datetime
from copy import deepcopy
import math
import random
a= [5,8,3,5,6,7,8]

bb= sorted(a, reverse=True)
tempdict = [0 for _ in range(len(a))]

for j, x in enumerate(bb):
    for i in range(len(a)):
        if  a[i]==x :
            tempdict[j] = i
            a[i]=None
            break
print tempdict
# for i in range(len(a)):
#     for j, x in enumerate(bb):
#         if  x == a[i]:
#             tempdict[j] = i
# print tempdict


# a= [5,8,3,5,6,7,8]