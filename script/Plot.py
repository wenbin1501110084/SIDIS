#!/usr/bin/env python3

from numpy import *
from sys import argv, exit
from os import path, mkdir
import h5py
from scipy import interpolate
import shutil
import pylab as pl
import numpy as np
HBARC = 0.197327053
from scipy.integrate import simps

aa=np.loadtxt("dSigma_dDeltaPhi_without_Sub")
xarr = []#concatenate((aa[:,0], 2* aa[-1:,0] - aa[:,0])) 
yarr = []
for ii in range(len(aa[:,0])):
    xarr.append(aa[ii,0])
    yarr.append(aa[ii,1])
    
for ii in range(len(aa[:,0])):
    xarr.append(2*pi - aa[len(aa[:,0])-ii-1,0])
    yarr.append(aa[len(aa[:,0])-ii-1,1])
xarr = np.array(xarr)
yarr = np.array(yarr)
integral_coherent = simps(yarr, xarr)
pl.figure(1002)
#pl.yscale('log')
pl.plot(xarr,yarr/integral_coherent,'r',label='CGC, W.O. Sub')
#pl.plot((pi-aa[:,0]) + pi, aa[:,1]/np.sum(aa[:,1]),'r',label='0.5$<p_T<5.0$ GeV')
pl.legend( loc='upper left' , fontsize=12)
#pl.xlim(0.31,3)
#pl.ylim(0,0.14)
pl.title('pp 200 GeV, 5 < $k_{\gamma \perp} $ < 7 GeV, 0.5<$P_{h \perp}$ < 1 GeV', fontsize=15)# give plot a title
pl.xlabel('$\Delta\phi$', fontsize=15)# make axis labels
pl.ylabel('$d\sigma/d\Delta\phi /\sigma$', fontsize=12)
pl.xticks(fontsize=10)
pl.yticks(fontsize=12)
#pl.show()# show the plot on the screen
pl.savefig('test.png', dpi=120)



