#!/usr/bin/env python

import sys

print(sys.argv[1])
f = open('results/' + sys.argv[1] + '.txt', 'r')

sum = 0
cnt = 0
for line in f:
    sum += float(line.replace('\n',''))
    cnt += 1
f.close()
print(sum / cnt)