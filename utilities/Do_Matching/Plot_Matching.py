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
from math import *
pl.figure(1002)

x0 = 0.01
aa = np.loadtxt("Matching_output")

diff = aa[:,3] / x0  + x0 * (aa[:,5]/0.0102 - aa[:,1]/0.0098)/(0.0102 - 0.0098)

pl.plot(aa[:,0],diff,'r',label='LHAPDF, x0')


for ii in range(1, len(aa[:,0])):
    if (diff[ii] < 0.0):
        print("Q0 ^2 in [GeV^2] = ", aa[ii,0])
        math_index = ii
        break

print("Rp in fm =  ", (aa[math_index,3] / aa[math_index,4] *HBARC * HBARC)**0.5)
pl.legend( loc='upper left' , fontsize=12)
#pl.xlim(0.31,2*pi-0.31)
#pl.ylim(0,0.14)
#pl.title('pp 200 GeV, 5 < $k_{\gamma \perp} $ < 7 GeV, 0.5<$P_{h \perp}$ < 1 GeV', fontsize=15)# give plot a title
pl.xlabel('$Q^{2}$', fontsize=15)# make axis labels
pl.ylabel('$xf$', fontsize=12)
pl.xticks(fontsize=10)
pl.yticks(fontsize=12)
#pl.show()# show the plot on the screen
pl.savefig('test_matching_1.png', dpi=120)





