# from matplotlib import pyplot as plt
# from scipy.special import softmax
# from itertools import product
import numpy as np
# from utils import *
# from model import *
# import time
# import seaborn as sns
from glob import glob as glob
import pandas as pd

gsname = 'gs1102'
dfpathL = glob('data/%s/*'%gsname)
dfL = []
for p in dfpathL[:100]:
  try:
    df_ = pd.read_csv(p)
    dfL.append(df_)
  except:
    print('error',p)
    continue
datadf = pd.concat(dfL).drop(columns='Unnamed: 0',)
# datadf ## full data
datadf.to_csv('data/%s-datadf.csv'%gsname)


### 
datadf.columns
paramL = ['concentration', 'stickiness_wi', 'stickiness_bt',
       'sparsity', 'pvar', 'lrate', 'lratep', 'decay_rate']

## compute summary df
groupvars = paramL 
gsdf_group = datadf.groupby(groupvars)
dfL = []
for params_i,df_i in gsdf_group:
  dataD = {**dict(zip(groupvars,params_i))}
  ## loop conditions (BIEML)
  for cond_i,df_c in df_i.groupby('cond'):
    ## compute metrics
    testacc = np.mean(df_c.acc[-40:])
    ## populate dataD
    dataD['testacc-%s'%cond_i[0]] = testacc
  # 
  dfL.append(pd.DataFrame(index=[0],data=dataD))
gsdf = pd.concat(dfL)


gsdf.to_csv('data/%s-summdf.csv'%gsname)