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
from scipy.integrate import simps
from scipy.optimize import curve_fit
#from math import *
from scipy.integrate import simps

def model_func2d(Q2,beta, Lambda2 ):
    Q2 = (Q2)
    return (1/(beta*log(Q2/Lambda2)))

bounds = ([0.1 ,0.0], [2, 1.0])  # Lower bounds, then upper bounds for each parameter

pl.figure(1002)
pl.yscale('log')
pl.xscale('log')
ss = np.loadtxt("alphas_Q2")
initial_guesses = [0.722, 0.224]

predictions = model_func2d(ss[:,0], initial_guesses[0], initial_guesses[1])
pl.plot(ss[:,0], (predictions),'r',label='CGC, W.O. Sub')

pl.plot(ss[:,0],ss[:,1],'g+',label='CGC, W.o. Sud, JAM20')
#pl.xlim(0.31,2*pi-0.31)
#pl.ylim(0,0.14)
pl.title('pp 200 GeV, 5 < $k_{\gamma \perp} $ < 7 GeV, 0.5<$P_{h \perp}$ < 1 GeV', fontsize=15)# give plot a title
pl.xlabel('$\Delta\phi$', fontsize=15)# make axis labels
pl.ylabel('$d\sigma/d\Delta\phi /\sigma$', fontsize=12)
pl.xticks(fontsize=10)
pl.yticks(fontsize=12)
#pl.show()# show the plot on the screen
pl.savefig('test_fcompare.png', dpi=120)





pl.figure(10042)
pl.yscale('log')
pl.xscale('log')
ss = np.loadtxt("alphas_Q2")
initial_guesses = [0.722, 0.224]
params, covariance = curve_fit(model_func2d, (ss[:,0]), (ss[:, 1]), p0=initial_guesses,bounds =bounds)
print(params)


pl.plot(ss[:,0],ss[:,1],'g+',label='CGC, W.o. Sud, JAM20')
predictions = model_func2d(ss[:,0], params[0], params[1])
pl.plot(ss[:,0], (predictions),'r',label='CGC, W.O. Sub')

#pl.xlim(0.31,2*pi-0.31)
#pl.ylim(0,0.14)
pl.title('pp 200 GeV, 5 < $k_{\gamma \perp} $ < 7 GeV, 0.5<$P_{h \perp}$ < 1 GeV', fontsize=15)# give plot a title
pl.xlabel('$\Delta\phi$', fontsize=15)# make axis labels
pl.ylabel('$d\sigma/d\Delta\phi /\sigma$', fontsize=12)
pl.xticks(fontsize=10)
pl.yticks(fontsize=12)
#pl.show()# show the plot on the screen
pl.savefig('test_compare.png', dpi=120)





