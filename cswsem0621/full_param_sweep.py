import os
from matplotlib import pyplot as plt
from scipy.special import softmax
from itertools import product
import numpy as np
from utils import *
from model import *
import time
import seaborn as sns
sns.set_context('talk')


## timestamp and dir for saving
tstamp = time.perf_counter_ns()
FIGDIR = 'figures/fullsweep/%i/'%tstamp
os.mkdir(FIGDIR)
os.mkdir(FIGDIR+'single')

## number seeds
ns = 50

## default base params
expargs = {
  'condition':'blocked',
  'n_train':160,
  'n_test':40
}
schargs = {
   'concentration':3,
   'stickiness_wi':10,
   'stickiness_bt':10,
   'sparsity':1,
   'pvar': 0,
   'lrate':1,
   'lratep':.3,
   'decay_rate':1,
}
semargs = {
  'beta2':False,
  'skipt1':True,
  'ppd_allsch':False
}
args = {
    'sem':semargs,
    'sch':schargs,
    'exp':expargs
}
param_str = "-".join(["%s_%.3f"%(i,j) for i,j in schargs.items()])
param_str += "-"+"-".join(["%s_%.3f"%(i,j) for i,j in semargs.items()])
param_str


# In[4]:

param_sweepD = {
    'concentration':np.arange(1,5,.25),
    'stickiness_wi':np.arange(5,15,1),
    'stickiness_bt':np.arange(0.01,5,.5),
    'sparsity':np.arange(0.01,1,0.1),
    'pvar':np.arange(0,2,0.2),
    'lrate':np.arange(0.8,1.01,0.05),
    'lratep':np.arange(.01,.5,.05),
    'decay_rate':np.arange(0.98,1.000001,0.001)
}


# ### main

def pltsave_macc(macc,schargs=None,labL=['B','I','E','M','L'],close=True):
  """ 
  save accuracy of individual parameter setting 
  used in paramsearch loops
  """
  plt.figure(figsize=(10,4))
  ax=plt.gca()
  for idx in range(len(macc)):
    ax.plot(macc[idx],label=labL[idx])
  ax.axhline(0.5,c='k')
  plt.legend()
  param_str = "-".join(["%s_%.3f"%(i,j) for i,j in schargs.items()])
  plt.savefig(FIGDIR+'single/acc-%s.jpg'%(param_str))
  if close:
    plt.close('all')
  return None


# In[6]:


dfL = []
condL = ['blocked','interleaved','early','middle','late']


for p_name,p_vals in param_sweepD.items():
    print(p_name,p_vals)

    for idx,p_val in enumerate(p_vals):
      print(idx/len(p_vals))
      
      args['sch'][p_name] = p_val  
      exp_batch_data = run_batch_exp_curr(ns,args,condL)
      ## acc
      batch_acc = unpack_acc(exp_batch_data) # curr,seeds,trials
      mean_acc = batch_acc.mean(1)
      test_acc = mean_acc[:,-40:].mean(1) # curr  
      
      ## save traces of EML for each param setting
      pltsave_macc(mean_acc[2:],args['sch'],labL=['E','M','L'])
      
      ## record
      gsD = {
        **schargs,
        **dict(zip(condL,test_acc))
      }
      dfL.append(gsD)
      
    gsdf = pd.DataFrame(dfL)

    ## plot sweep 
    plt.figure(figsize=(20,10))
    ax = plt.gca()
    gsdf.plot(p_name,condL,ax=ax)
    ax.set_ylim(0.4,1.05)
    ax.set_ylabel('test acc')
    for i in np.arange(0.5,1.01,0.1):
      ax.axhline(i,c='k',lw=0.5)
    plt.title(param_str)
    plt.savefig(FIGDIR+'testacc-sweep_%s-default_%s-t%s.png'%(
      p_name,param_str,tstamp))
    plt.close('all')

