a=[1.,2.,3.]
#a=array(a)
b=[(b,c) for b in a for c in a ]
# [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (2.0, 1.0),
#(2.0, 2.0), (2.0, 3.0), (3.0, 1.0), (3.0, 2.0), (3.0, 3.0)]
#for u in b:print u[0],u[1],eucDist(u[0],u[1])
#1.0 1.0 0.0
#..
#3.0 3.0 0.0
b1=[[u,u] for u in b]
#[ eucDist(e[0],e[1]) for e in b1]

a=[1.,2.]
b=[5.,6.]
c=[[e,f] for e in a for f in b ]
#>>> c
# [[1.0, 5.0], [1.0, 6.0], [1.0, 7.0], [2.0, 5.0], [2.0, 6.0],
# [2.0, 7.0], [3.0, 5.0], [3.0, 6.0], [3.0, 7.0]]
#print eucDist(c[0],c[1])
# 1.0

'''
#a=3
print "aa=",aa,dir(aa)
for i in dir(aa):
    try:
        print aa,i
        setattr(testScript,"aa",3)
        print aa
    except:
        pass
   '''

from numpy import *

class kovar(arraytype):
    def __init__(self):
        arraytype.__init__(self)

        