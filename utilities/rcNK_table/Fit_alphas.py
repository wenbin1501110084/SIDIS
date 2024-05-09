#!/usr/bin/env python3

from numpy import *
from sys import argv, exit
from os import path, mkdir
import h5py
from scipy import interpolate
import shutil
import pylab as pl
import numpy as np
import math

HBARC = 0.197327053
from scipy.integrate import simps
from scipy.integrate import simps
from scipy.optimize import curve_fit
#from math import *
from scipy.integrate import simps

mub2 = 2.0


def model_func2d(Q2,beta, Lambda2 ):
    Q2 = (Q2)
    return (1/(beta*log(Q2/Lambda2)))

def model_func2c(Q2, A, B ):
    beta = 0.77630491
    Lambda = 0.05243429**0.5
    Q = Q2**0.5
    mub = mub2**0.5
    AA = 1/beta * (-2*A*log(Q/mub) + (B + 2*A*log(Q/Lambda)) * (log((log(Q/Lambda)) / (log(mub/Lambda)))))
    return AA
    
    
bounds = ([0.1 ,0.0], [2, 1.0])  # Lower bounds, then upper bounds for each parameter

initial_guesses = [0.722, 0.224]

pl.figure(10042)
pl.yscale('log')
pl.xscale('log')
ss = np.loadtxt("alphas_Q2")
initial_guesses = [0.722, 0.224]
params, covariance = curve_fit(model_func2d, (ss[:,0]), (ss[:, 1]), p0=initial_guesses,bounds =bounds)


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


pl.figure(1003242)
#pl.yscale('log')
#pl.xscale('log')
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

CA = 3.
CF = 4./3.
Q2 = np.arange(mub2+1, 100, 2)

SSud = []
for iq in range(len(Q2)):
    mu2 = np.arange(mub2, Q2[iq], 0.1)
    Ssud = 0.0
    alphas = model_func2d(mu2, params[0], params[1])
    A = alphas/math.pi * (CF + CA/2)
    B = -alphas/math.pi *1.5*CF
    stem = 1/mu2 * (A*log(Q2[iq]/mu2) + B)
    integral_coherent = simps(stem, mu2)
    SSud.append(integral_coherent)
SSud = np.array(SSud)

pl.figure(1042)

pl.plot(Q2,SSud,'g+',label='CGC, W.o. Sud, JAM20')

params, covariance = curve_fit(model_func2c, Q2, SSud)#, p0=initial_guesses)
print(params)


predictions = model_func2c(Q2, params[0], params[1])
pl.plot(Q2, (predictions),'r',label='CGC, W.O. Sub')

#pl.xlim(0.31,2*pi-0.31)
#pl.ylim(0,0.14)
pl.title('pp 200 GeV, 5 < $k_{\gamma \perp} $ < 7 GeV, 0.5<$P_{h \perp}$ < 1 GeV', fontsize=15)# give plot a title
pl.xlabel('$\Delta\phi$', fontsize=15)# make axis labels
pl.ylabel('$d\sigma/d\Delta\phi /\sigma$', fontsize=12)
pl.xticks(fontsize=10)
pl.yticks(fontsize=12)
#pl.show()# show the plot on the screen
pl.savefig('SSud_Q2.png', dpi=120)




























