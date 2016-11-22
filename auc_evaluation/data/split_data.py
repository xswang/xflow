import sys
import random

f=open(sys.argv[1], 'r')
num = int(sys.argv[2])
ftype = sys.argv[3]
names = []
for i in xrange(num):
    name = ftype + '-0000' + str(i)
    names.append(name)
f1 = open(names[0], 'w')
f2 = open(names[1], 'w')
f3 = open(names[2], 'w')

for line in f:
    v = random.random()
    if 0 <= v and v < 1.0/num:
        f1.write(line.strip())
        f1.write('\n')
    elif 1.0 / num <= v and v < 2 * 1.0/num:
        f2.write(line.strip())
        f2.write('\n')
    elif 2 * 1.0/num <= v and v < 1.0:
        f3.write(line.strip())
        f3.write('\n')
f1.close()
f2.close()
f3.close()
