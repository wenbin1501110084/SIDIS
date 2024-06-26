#!/usr/bin/env python3

from numpy import *
from sys import argv, exit
from os import path, mkdir
#from math import *
import math
from scipy import interpolate
import shutil
import pylab as pl
import numpy as np
HBARC = 0.197327053
from scipy.integrate import simps
from scipy.optimize import curve_fit
#from math import *
from scipy.integrate import simps

def linear_extrapolation(x1, y1, x2, y2, x_new):
    # Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)
    
    # Calculate the y-intercept (c)
    c = y1 - m * x1
    
    # Calculate the new y value for x_new
    y_new = m * x_new + c
    
    return y_new

def model_func2c(kT, A,C, D ):
    AA =  A*kT**0.5  + C*kT +D
    return AA
    
    
aa=np.loadtxt("Proton_MvGamma_F_table_K_A_all")
#bb = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/Dijet/photon_hadron/Paul_table/rcNK_table_K_A_all_rcBK-MVgamma_large_nucleus")
mmm = np.array([])
for iy in range(80):
    exec("pl.figure({})".format(iy+159))
    #pl.plot(bb[int(iy*700):int(iy*700+700),1],bb[int(iy*700):int(iy*700+700),2]* bb[int(iy*700):int(iy*700+700),1] * bb[int(iy*700):int(iy*700+700),1]/2/math.pi/math.pi/2/math.pi/2,'c')
    
    cc = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/Dijet/photon_hadron/FTtable/FT_table_K_pMVgamma_Paul_proton_{}_{}.txt".format(int(iy), int(iy+1)))
    cc[0,1] = 0.00001
    x1 = cc[2,1]
    x2 = cc[3,1]
    y1 = -cc[2,3]* cc[2,1] * cc[2,1]/2.25
    y2 = -cc[3,3]* cc[3,1] * cc[3,1]/2.25
    print(y1, y2)
    temp = linear_extrapolation(math.log(x1), math.log(y1),
                                math.log(x2), math.log(y2), math.log(cc[0,1]))
    cc[0,3] = -math.exp(temp)*2.25/cc[0,1] /cc[0,1] 
    
    
    aa[int(iy*700),1] = 0.00001
    temp = linear_extrapolation(math.log(aa[int(iy*700+2),1]), math.log(aa[int(iy*700+2),2]),
                                math.log(aa[int(iy*700+3),1]), math.log(aa[int(iy*700+3),2]), math.log(aa[int(iy*700),1]))
    aa[int(iy*700),2] = math.exp(temp)
    init2 = 100 + iy*5
    if (iy < 90):
        ini = 59 
        fin = 190 
        if (iy > 30):
            ini = 59 + iy*2 -30
            fin = 190 + iy*2 -30
        pp = cc[int(0*700+init2):int(0*700+700),1]
        rr = cc[int(0*700+ini):int(0*700+fin),1]
        print(rr)
        FF = -cc[int(0*700+ini):int(0*700+fin),3]* cc[int(0*700+ini):int(0*700+fin),1] * cc[int(0*700+ini):int(0*700+fin),1]/2.25
        params, covariance = curve_fit(model_func2c, np.log(rr), np.log(FF))#, p0=initial_guesses)
        predictions = model_func2c(np.log(pp), params[0], params[1], params[2])
        cc[int(0*700+init2):int(0*700+700),3] = -np.exp(predictions)*2.25/(pp) /(pp) 
        
        init22=50
        #ini = 50
        #fin = 200
        pp = aa[int(iy*700+init22):int(iy*700+700),1]
        rr = aa[int(iy*700+ini):int(iy*700+fin),1]
        FF = aa[int(iy*700+ini):int(iy*700+fin),3]
        params, covariance = curve_fit(model_func2c, np.log(rr), np.log(FF))#, p0=initial_guesses)
        predictions = model_func2c(np.log(pp), params[0], params[1], params[2])
        aa[int(iy*700+init22):int(iy*700+700),2] = np.exp(predictions)
        
        #ini = 50
        #fin = 200
        init22=50
        pp = aa[int(iy*700+init22):int(iy*700+700),1]
        rr = aa[int(iy*700+ini):int(iy*700+fin),1]
        FF = aa[int(iy*700+ini):int(iy*700+fin),3]
        params, covariance = curve_fit(model_func2c, np.log(rr), np.log(FF))#, p0=initial_guesses)
        predictions = model_func2c(np.log(pp), params[0], params[1], params[2])
        aa[int(iy*700+init22):int(iy*700+700),3] = np.exp(predictions)
        
        #ini = 50
        #fin = 200
        pp = aa[int(iy*700+init2):int(iy*700+700),1]
        rr = aa[int(iy*700+ini):int(iy*700+fin),1]
        FF = aa[int(iy*700+ini):int(iy*700+fin),4]
        params, covariance = curve_fit(model_func2c, np.log(rr), np.log(FF))#, p0=initial_guesses)
        predictions = model_func2c(np.log(pp), params[0], params[1], params[2])
        aa[int(iy*700+init2):int(iy*700+700),4] = np.exp(predictions)
        
        #ini = 50
        #fin = 200
        pp = aa[int(iy*700+init2):int(iy*700+700),1]
        rr = aa[int(iy*700+ini):int(iy*700+fin),1]
        FF = aa[int(iy*700+ini):int(iy*700+fin),-1]
        params, covariance = curve_fit(model_func2c, np.log(rr), np.log(FF))#, p0=initial_guesses)
        predictions = model_func2c(np.log(pp), params[0], params[1], params[2])
        aa[int(iy*700+init2):int(iy*700+700),-1] = np.exp(predictions)
        
        #ini = 50
        #fin = 200
        pp = aa[int(iy*700+init2):int(iy*700+700),1]
        rr = aa[int(iy*700+ini):int(iy*700+fin),1]
        FF = aa[int(iy*700+ini):int(iy*700+fin),6]
        params, covariance = curve_fit(model_func2c, np.log(rr), np.log(FF))#, p0=initial_guesses)
        predictions = model_func2c(np.log(pp), params[0], params[1], params[2])
        aa[int(iy*700+init2):int(iy*700+700),6] = np.exp(predictions)
        
    pl.plot(cc[:,1],-cc[:,3]* cc[:,1] * cc[:,1]/2.25,'y--',label='$\\alpha_s*F_{adj}$')
    mmm = np.concatenate((mmm, -cc[:,3]* cc[:,1] * cc[:,1]/2.25))
    pl.plot(aa[int(iy*700):int(iy*700+700),1],aa[int(iy*700):int(iy*700+700),2],'k--',label='$\\alpha_s*F^{1}_{qg}$')
    pl.plot(aa[int(iy*700):int(iy*700+700),1],aa[int(iy*700):int(iy*700+700),3],'b--',label='$\\alpha_s*F^{2}_{qg}$')
    pl.plot(aa[int(iy*700):int(iy*700+700),1],aa[int(iy*700):int(iy*700+700),4],'r--',label='$\\alpha_s*F^{1}_{gg}$')
    
    #pl.plot(aa[int(iy*700):int(iy*700+700),1], -aa[int(iy*700):int(iy*700+700),7],'m--',label='$\\alpha_s*F_{adj}$')
    pl.plot(aa[int(iy*700):int(iy*700+700),1],aa[int(iy*700):int(iy*700+700),-1],'c--',label='$\\alpha_s*F^{6}_{gg}$')
    pl.plot(aa[int(iy*700):int(iy*700+700),1],aa[int(iy*700):int(iy*700+700),6],'g--',label='$\\alpha_s*F^{3}_{gg}$')
    
    
    pl.yscale('log')
    pl.xscale('log')
    pl.legend( loc='upper right' , fontsize=12)
    pl.xlim(1e-2,100)
    pl.ylim(1e-6,1)
    print(iy*0.2)
    pl.title('y = {}, MV gamma Paul, proton'.format(iy*0.2), fontsize=15)# give plot a title
    pl.xlabel('$k_{T}$ [GeV]', fontsize=15)# make axis labels
    pl.ylabel('$\\alpha_s F/S_{\perp}$', fontsize=12)
    pl.xticks(fontsize=10)
    pl.yticks(fontsize=12)
    #pl.show()# show the plot on the screen
    pl.savefig('PNG_proton/F_{}.png'.format(iy), dpi=120)

np.savetxt("Regularged_FT_proton_MVgamma_Paul",
        # Y, pT, F1qg, F2qg, F1gg, F3gg, F6gg, Fadj
        np.array([aa[:,0], aa[:,1], aa[:,2], aa[:,3], aa[:,4], aa[:,6], aa[:,-1], mmm]).transpose(),
        fmt="%.6e", delimiter="  ")
        

    
    
    
    
    
    
    
    
    
