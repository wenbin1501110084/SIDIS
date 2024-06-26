#!/usr/bin/env python3
import numpy as np
#import collections
#from enum import Enum
import math
import pylab as pl
#from collections import Counter
#import pandas as pd
from numpy import *
from scipy import optimize, special
from scipy import *
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp 
from sys import argv, exit 
hbarc = 0.197
ll = 39
for ievent in range(1):
#for ievent in range(11):
    aa = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma_all_RHIC")
    a0 = aa[:,0][ aa[:,2] >0.0
    ]
    a1 = aa[:,1][ aa[:,2] >0.0
    ]
    a2 = aa[:,2][ aa[:,2] >0.0
    ]
    a3 = aa[:,3][ aa[:,2] >0.0
    ]
    a4 = aa[:,4][ aa[:,2] >0.0
    ]
    aa = np.column_stack((a0, a1, a2, a3, a4))
    
    xx = aa[:,1]
    yy = aa[:,0]
    x = xx[0:ll]
    y = yy[0::ll]
    xg = aa[:,3] / aa[:,2]
    allxg = []
    for ikk in range(int(len(xx)/len(x))):
        mid = xg[ikk*ll:ikk*ll+ll]
        allxg.append(mid)
    allxg = np.array(allxg)
    print(allxg)
    print(x, y)
    X, Y = np.meshgrid(x, y)
    exec("pl.figure({})".format(ievent))
    #print(np.max(allxg))
    levels = np.linspace(0.0, 0.25, 100)
    levels2 = np.linspace(0.0, 0.25, 5)
    #levels = np.linspace(np.min(T00),np.max(T00),50)
    cs = pl.contourf(X, Y, allxg, levels=levels,colorscale='Electric', fill=False,cmap=pl.cm.jet)#cmap="Blues")
    cbar = pl.colorbar(cs)

    # Adjust colorbar ticks format
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(levels2)
    cbar.ax.set_yticklabels(['{:.2f}'.format(val) for val in levels2])
    
    
    contour = pl.contour(X, Y, allxg, levels=[0.01], colors='white')

    # Add contour label for the specified level
    pl.clabel(contour, fmt='%1.2f', inline=True, fontsize=20)

    #pl.xlim(-2.5,2.5)
    #pl.ylim(-2.5,2.5)
    #pl.legend( loc='upper right' , fontsize=10)
    pl.xlabel('$P_{hT}$ [GeV]', fontsize=14)# make axis labels
    pl.ylabel('$k_{T, \gamma}$ [GeV]', fontsize=15)
    pl.xticks(fontsize=14)
    pl.yticks(fontsize=15)
    pl.title('<$x_{g}>$ in p-p 200 GeV, $|\eta_{\gamma}|<0.35, |\eta_{h}|<0.35$', fontsize=14)# give plot a title
    pl.savefig("xg_RHIC.png".format(ievent))
    
    
    aa = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma_all_LHC_ALICE")
    a0 = aa[:,0][ aa[:,2] >0.0
    ]
    a1 = aa[:,1][ aa[:,2] >0.0
    ]
    a2 = aa[:,2][ aa[:,2] >0.0
    ]
    a3 = aa[:,3][ aa[:,2] >0.0
    ]
    a4 = aa[:,4][ aa[:,2] >0.0
    ]
    aa = np.column_stack((a0, a1, a2, a3, a4))
    xx = aa[:,1]
    yy = aa[:,0]
    x = xx[0:ll]
    y = yy[0::ll]
    xg = aa[:,3] / aa[:,2]
    allxg = []
    for ikk in range(int(len(xx)/len(x))):
        mid = xg[ikk*ll:ikk*ll+ll]
        allxg.append(mid)
    allxg = np.array(allxg)
    print(allxg)
    print(x, y)
    X, Y = np.meshgrid(x, y)
    exec("pl.figure({})".format(1235))
    print("Max = ", np.max(allxg))
    levels = np.linspace(0.0, 0.025, 100)
    levels2 = np.linspace(0.0, 0.025, 10)
    #levels = np.linspace(np.min(T00),np.max(T00),50)
    cs = pl.contourf(X, Y, allxg, levels=levels,colorscale='Electric', fill=False,cmap=pl.cm.jet)#cmap="Blues")
    cbar = pl.colorbar(cs)

    # Adjust colorbar ticks format
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(levels2)
    cbar.ax.set_yticklabels(['{:.4f}'.format(val) for val in levels2])
    
    
    contour = pl.contour(X, Y, allxg, levels=[0.01], colors='white')

    # Add contour label for the specified level
    pl.clabel(contour, fmt='%1.2f', inline=True, fontsize=20)

    #pl.xlim(-2.5,2.5)
    #pl.ylim(-2.5,2.5)
    #pl.legend( loc='upper right' , fontsize=10)
    pl.xlabel('$P_{hT}$ [GeV]', fontsize=14)# make axis labels
    pl.ylabel('$k_{T, \gamma}$ [GeV]', fontsize=15)
    pl.xticks(fontsize=14)
    pl.yticks(fontsize=15)
    pl.title('<$x_{g}>$ in p-p 5.02 TeV, $|\eta_{\gamma}|<0.67, |\eta_{h}|<0.8$', fontsize=14)# give plot a title
    pl.savefig("xg_LHC.png".format(ievent))
    
    
    aa = np.loadtxt("Xg_kT_dis_with_Sub_FBT_MVgamma_LHCb")
    a0 = aa[:,0][ aa[:,2] >0.0
    ]
    a1 = aa[:,1][ aa[:,2] >0.0
    ]
    a2 = aa[:,2][ aa[:,2] >0.0
    ]
    a3 = aa[:,3][ aa[:,2] >0.0
    ]
    a4 = aa[:,4][ aa[:,2] >0.0
    ]
    aa = np.column_stack((a0, a1, a2, a3, a4))
    xx = aa[:,1]
    yy = aa[:,0]
    x = xx[0:ll]
    y = yy[0::ll]
    print(len(x), len(y))
    xg = aa[:,3] / aa[:,2]
    allxg = []
    for ikk in range(int(len(xx)/len(x))):
        mid = xg[ikk*ll:ikk*ll+ll]
        allxg.append(mid)
    allxg = np.array(allxg)
    print(allxg)
    print(x, y)
    X, Y = np.meshgrid(x, y)
    exec("pl.figure({})".format(12345))
    print("Max = ", np.max(allxg))
    levels = np.linspace(0.0, 0.0015, 100)
    levels2 = np.linspace(0.0, 0.0015, 10)
    #levels = np.linspace(np.min(T00),np.max(T00),50)
    cs = pl.contourf(X, Y, allxg, levels=levels,colorscale='Electric', fill=False,cmap=pl.cm.jet)#cmap="Blues")
    cbar = pl.colorbar(cs)

    # Adjust colorbar ticks format
    cbar.ax.tick_params(labelsize=14)
    cbar.set_ticks(levels2)
    cbar.ax.set_yticklabels(['{:.4f}'.format(val) for val in levels2])
    
    
    contour = pl.contour(X, Y, allxg, levels=[0.01], colors='white')

    # Add contour label for the specified level
    pl.clabel(contour, fmt='%1.2f', inline=True, fontsize=20)

    #pl.xlim(-2.5,2.5)
    #pl.ylim(-2.5,2.5)
    #pl.legend( loc='upper right' , fontsize=10)
    pl.xlabel('$P_{hT}$ [GeV]', fontsize=14)# make axis labels
    pl.ylabel('$k_{T, \gamma}$ [GeV]', fontsize=15)
    pl.xticks(fontsize=14)
    pl.yticks(fontsize=15)
    pl.title('<$x_{g}>$ in p-p 5.02 TeV, $1.5<\eta_{\gamma}, \eta_{h}<4.0$', fontsize=14)# give plot a title
    pl.savefig("xg_LHCb.png".format(ievent))
    
    
    
    
    
    
