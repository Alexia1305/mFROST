# add current directory

import numpy as np
import pylab as pl
#import os, sys
#sys.path.append("/Users/vgligorijevic/Projects/NF-CCE/data/mostafavi/")

from snmtf import SNMTF
from csnmtf import CSNMTF
from preprocessing import nets_from_mat, mltplx_from_mat, net_normalize
from cluster import nmf_clust, spect_clust, clust_eval

from csnmtf_R import CSNMTF_algo
## create symmetric matrix (test)
#X = np.mat(np.random.rand(10, 10))
#X = 0.5*(X + X.T)
#X = X - np.diag(np.diag(X))

## load Mostafavi data
#genes, _, Nets, _ = nets_from_mat("../data/mostafavi/human_and_go.mat")
#Nets = Nets[:3]

## load Dong data
Nets, ground_idx = mltplx_from_mat("NFCCE/data/nets/mit.mat", 'mit')


#objSNMF = SNMF(Nets[1], 50, init='rnda', displ='true')
#res_snmf = objSNMF.fit()

nmf_idx=CSNMTF_algo(Nets,max(ground_idx),alpha=0.05)
print(clust_eval(ground_idx, nmf_idx))
