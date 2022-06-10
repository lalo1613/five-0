import os, time

os.environ['R_HOME'] = r'C:\Program Files\R\R-3.5.2'
os.environ['R_USER'] = r'C:\Users\omri_\PycharmProjects\Five-0\venv\Lib\site-packages\rpy2'

import rpy2.robjects as robjects

robjects.r.source("handOdds.R")

r_flush = robjects.r('flushOdds')

#print(r_flush(2,7,30))

matr1 = [1., 3., 1., 3., 0.2, 1.5, 0.5, 1.3]
matr2 = [1., 3., 1., 3., 0.2, 1.5, 0.5, 1.3, 0.5, 4.6]

m1 = robjects.r.matrix(robjects.FloatVector(matr1), nrow=int(len(matr1)/2), byrow=True)
m2 = robjects.r.matrix(robjects.FloatVector(matr2), nrow=int(len(matr2)/2), byrow=True)

r_foo = robjects.r('foo123')
a = r_foo(m1)

#a = [-1 for y in range(5)] + [-1 for x in range(5)]

print(m1)

#start = time.time()

#a=robjects.r("unname(system.time(replicate(100,straightOdds(iterations=100,amounts = rep("+str(c)+",9))))[3])")[0]

#b=robjects.r("unname(system.time(replicate(1000,flushOdds(1,10,41)))[3])")[0]
#b=robjects.r("b=flushOdds(2,10,34)")

#end = time.time()

#print((end - start) - a - b)