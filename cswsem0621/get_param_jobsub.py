import sys
import itertools
import numpy as np

"""
controls the gridsearch parameter space
given an index, return parameter string
""" 

param_set_idx = int(sys.argv[1])

c = [0] # concentration sweep within gs script
stwi = [4000,5000] # stickiness_wi
stbt = [8,10] # stickiness_bt
sp = np.arange(0.01,0.5,0.025) # sparsity
pvar = [1,2] # pvar
lrate = [0.8,1] # lrate
lratep = [0.8,0.9,1] # lratep
decay = [0.99,1] # decay_rate

itrprod = itertools.product(
    c,stwi,stbt,sp,pvar,lrate,lratep,decay
)

gsize = len(c)*len(stwi)*len(stbt)*len(sp)*\
len(pvar)*len(lrate)*len(lratep)*len(decay)

print('grid size',gsize,'COMMENT OUT BEFORE RUNNING')

for idx,paramL in enumerate(itrprod):
  if idx == param_set_idx:
    print(" ".join([str(i) for i in paramL]))
    break


