import sys
import itertools
import numpy as np

"""
controls the gridsearch parameter space
given an index, return parameter string
""" 

param_set_idx = int(sys.argv[1])

c = np.arange(2,15,0.25) # concentration sweep within gs script
stwi = [5000] # stickiness_wi
stbt = [0] # stickiness_bt
sp = np.arange(0.01,1.2,0.025) # sparsity
pvar = [2] # pvar
lrate = [1] # lrate
lratep = [1] # lratep
decay = [1] # decay_rate

itrprod = itertools.product(
    c,stwi,stbt,sp,pvar,lrate,lratep,decay
)

gsize = len(c)*len(stwi)*len(stbt)*len(sp)*\
len(pvar)*len(lrate)*len(lratep)*len(decay)

# print('grid size',gsize,'COMMENT OUT BEFORE RUNNING')

for idx,paramL in enumerate(itrprod):
  if idx == param_set_idx:
    print(" ".join([str(i) for i in paramL]))
    break


