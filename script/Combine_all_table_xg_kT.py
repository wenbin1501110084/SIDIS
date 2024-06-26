#!/usr/bin/env python3

from numpy import *
from sys import argv, exit
from os import path, mkdir
#from math import *
import math
from scipy import interpolate
import shutil
import numpy as np
HBARC = 0.197327053
from scipy.integrate import simps
from scipy.optimize import curve_fit
#from math import *
from scipy.integrate import simps
name ='12_15_5_10'
#aa = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__11_0_1_0_5.txt")
aa = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__11_0_1_1_2.txt")
aa = aa.reshape((1, len(aa)))  
for ik in range(1, 25):
    bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__11_0_1_{}_{}.txt".format(int(ik), int(ik+1)))
    bb = bb.reshape((1, len(bb)))  
    aa = concatenate((aa, bb))
    #print(aa.shape)
for jj in range(20):
    for kk in range(5):
        if (jj == 0 & kk ==0):
            continue
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__11_{}_{}_{}_{}.txt".format(int(jj), int(jj+1), int(kk*5), int(kk*5+5)))
        aa = concatenate((aa, bb))

    for kk in range(2):
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__12_{}_{}_{}_{}.txt".format(int(jj), int(jj+1), int(kk*5), int(kk*5+5)))
        aa = concatenate((aa, bb))
    
    for kk in range(1):
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__13_{}_{}_0_5.txt".format(int(jj), int(jj+1)))
        aa = concatenate((aa, bb))

for ik in range(1, 25):
    bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__21_0_1_{}_{}.txt".format(int(ik), int(ik+1)))
    bb = bb.reshape((1, len(bb)))
    aa = concatenate((aa, bb))


for jj in range(10):
    for kk in range(5):
        if (jj ==0 & kk==0):
             continue
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__21_{}_{}_{}_{}.txt".format(int(jj), int(jj+1), int(kk*5), int(kk*5+5)))
        aa = concatenate((aa, bb))

    for kk in range(2):
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__22_{}_{}_{}_{}.txt".format(int(jj), int(jj+1), int(kk*5), int(kk*5+5)))
        aa = concatenate((aa, bb))
    
    for kk in range(1):
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__23_{}_{}_0_5.txt".format(int(jj), int(jj+1)))
        aa = concatenate((aa, bb))
        

for ik in range(1, 25):
    bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__31_0_1_{}_{}.txt".format(int(ik), int(ik+1)))
    bb = bb.reshape((1, len(bb)))
    aa = concatenate((aa, bb))


for jj in range(5):
    for kk in range(5):
        if (jj ==0 & kk==0):
             continue
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__31_{}_{}_{}_{}.txt".format(int(jj), int(jj+1), int(kk*5), int(kk*5+5)))
        #try:
        aa = concatenate((aa, bb))
        #except ValueError as e:
        #    print(jj, kk)

    for kk in range(2):
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__32_{}_{}_{}_{}.txt".format(int(jj), int(jj+1), int(kk*5), int(kk*5+5)))
        aa = concatenate((aa, bb))
    
    for kk in range(1):
        bb = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma__33_{}_{}_0_5.txt".format(int(jj), int(jj+1)))
        aa = concatenate((aa, bb))
        
        
np.savetxt("Xg_kT_dis_with_Sub_FBT_MVgamma_all", aa, fmt="%.6e", delimiter="  ")

