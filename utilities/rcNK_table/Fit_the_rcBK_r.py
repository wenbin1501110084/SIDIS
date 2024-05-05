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

pl.figure(1002)

def model_func2d(r, A, B, C, D, E, F):
    return 1 - A * exp(-r**B * C + D) + E*r**F

def model_func2db(r, A, B, C, D):
    E = -1.992483397709988093e-03
    F = 7.277427057067667702e-02
    return 1 - A * exp(-r**B * C + D) + E*r**F
    
    
aa=np.loadtxt("rcBK_MVe_Heikki_table.txt")
YY = aa[:,0]
yy = YY[0::400]
#print(yy)
rry16 = aa[:,1][ aa[:,0] == 16
]
pl.plot(aa[0:400,1],aa[0:400, 2],'r+',label='CGC, W.O. Sub')
params, covariance = curve_fit(model_func2d, rry16, aa[0:400, 2])
print("y ={} ".format(yy[0]), params)
# Generate predictions using the fitted parameters
predictions = model_func2d(rry16, params[0], params[1], params[2], params[3], params[4], params[5])
pl.plot(rry16,predictions,'r',label='CGC, W.O. Sub')

rry16 = aa[:,1][ aa[:,0] == yy[-3]
]
NN16 = aa[:,2][ aa[:,0] == yy[-3]
]
iy = -3
gass = np.loadtxt("All_fit_paras")
initial_guesses = (gass[iy, 1], gass[iy, 2], gass[iy, 3], gass[iy, 4], gass[iy, 5], gass[iy, 6]) 
initial_guesses2 = (gass[iy, 1], gass[iy, 2], gass[iy, 3], gass[iy, 4]) 

params, covariance = curve_fit(model_func2d, rry16, NN16, p0=initial_guesses)
pl.plot(rry16,NN16,'b+',label='CGC, W.O. Sub')
predictions = model_func2d(rry16, params[0], params[1], params[2], params[3], params[4], params[5])
pl.plot(rry16,predictions,'b--',label='CGC, Wff.O. Sub')

params, covariance = curve_fit(model_func2db, rry16, NN16, p0=initial_guesses2)
print("params =============== ", params, yy[-3])
predictions = model_func2db(rry16, params[0], params[1], params[2], params[3])
pl.plot(rry16,predictions,'k',label='CGC, W.O. Sub')


#pl.plot(rry16,NN16,'bo',label='CGC, W.O. Sub')
#pl.plot((pi-aa[:,0]) + pi, aa[:,1]/np.sum(aa[:,1]),'r',label='0.5$<p_T<5.0$ GeV')
pl.legend( loc='lower right' , fontsize=12)
pl.xlim(0.,20)
#pl.ylim(0,0.14)
pl.title('pp 200 GeV, 5 < $k_{\gamma \perp} $ < 7 GeV, 0.5<$P_{h \perp}$ < 1 GeV', fontsize=15)# give plot a title
pl.xlabel('$\Delta\phi$', fontsize=15)# make axis labels
pl.ylabel('$d\sigma/d\Delta\phi /\sigma$', fontsize=12)
pl.xticks(fontsize=10)
pl.yticks(fontsize=12)
#pl.show()# show the plot on the screen
pl.savefig('testff.png', dpi=120)


aa=np.loadtxt("rcBK_MVe_Heikki_table.txt")
gass = np.loadtxt("All_fit_paras")
All_fit_paras = []
integrated_Nr = []
for iy in range(len(yy)):
    exec("pl.figure({})".format(iy+100e3))
    rry16 = aa[:,1][ (aa[:,0] == yy[iy]) & (aa[:,1] <200)
    ]
    NN16 = aa[:,2][ (aa[:,0] == yy[iy]) & (aa[:,1] <200)
    ]
    
    rry20 = aa[:,1][ (aa[:,0] == yy[iy])
    ]
    NN20 = aa[:,2][ (aa[:,0] == yy[iy]) 
    ]
    print(len(rry20), len(NN20))
    integral_coherent = simps((1-NN20) * rry20 * 2 * math.pi, rry20)
    integrated_Nr.append(integral_coherent)
    initial_guesses = (gass[iy, 1], gass[iy, 2], gass[iy, 3], gass[iy, 4], gass[iy, 5], gass[iy, 6]) 
    params, covariance = curve_fit(model_func2d, rry16, NN16, p0=initial_guesses)
    print("y ={} ".format(yy[iy]), params)
    mid=[]
    mid.append(yy[iy])
    for ip in range(len(params)):
        mid.append(params[ip])
    All_fit_paras.append(mid)
    pl.plot(rry20,NN20,'b+',label='CGC, W.O. Sub')
    predictions = model_func2d(rry20, params[0], params[1], params[2], params[3], params[4], params[5])
    pl.plot(rry20,predictions,'b',label='CGC, W.O. Sub')

    #pl.plot(rry16,NN16,'bo',label='CGC, W.O. Sub')
    #pl.plot((pi-aa[:,0]) + pi, aa[:,1]/np.sum(aa[:,1]),'r',label='0.5$<p_T<5.0$ GeV')
    #pl.legend( loc='lower right' , fontsize=12)
    pl.xlim(0.,20)
    #pl.ylim(0,0.14)
    pl.title('y = {}'.format(yy[iy]), fontsize=15)# give plot a title
    pl.xlabel('$\Delta\phi$', fontsize=15)# make axis labels
    pl.ylabel('$d\sigma/d\Delta\phi /\sigma$', fontsize=12)
    pl.xticks(fontsize=10)
    pl.yticks(fontsize=12)
    #pl.show()# show the plot on the screen
    pl.savefig('test_all_{}.png'.format(iy), dpi=120)

integrated_Nr = np.array(integrated_Nr)
np.savetxt("integrated_Nr", integrated_Nr)
All_fit_paras = np.array(All_fit_paras)
np.savetxt("All_fit_paras", All_fit_paras)
All_fit_paras = np.loadtxt("All_fit_paras_fixed")
for iy in range(6):
    exec("pl.figure({})".format(iy+10e3))
    pl.plot(All_fit_paras[:,0], All_fit_paras[:,iy+1],'b')
    #pl.plot(rry16,NN16,'bo',label='CGC, W.O. Sub')
    #pl.plot((pi-aa[:,0]) + pi, aa[:,1]/np.sum(aa[:,1]),'r',label='0.5$<p_T<5.0$ GeV')
    #pl.legend( loc='lower right' , fontsize=12)
    #pl.xlim(0.,20)
    #pl.ylim(0,0.14)
    pl.title('y = {}'.format(yy[iy]), fontsize=15)# give plot a title
    pl.xlabel('$\Delta\phi$', fontsize=15)# make axis labels
    pl.ylabel('$d\sigma/d\Delta\phi /\sigma$', fontsize=12)
    pl.xticks(fontsize=10)
    pl.yticks(fontsize=12)
    #pl.show()# show the plot on the screen
    pl.savefig('para_{}.png'.format(iy), dpi=120)
    

"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read 3D data from file
data = np.loadtxt('rcBK_MVe_Heikki_table.txt')  # Replace 'your_data_file.txt' with the path to your data file
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

def model_func(data, A, B, C, D, F):
    Y, r = data
    print(Y)
    GG = F * log(Y)
    return 1 - A * exp(-r*r * GG * B) + C * r**D

def model_func_d(Y, r, A, B, C, D, F):
    GG = F * log(Y)
    return 1 - A * exp(-r*r * GG * B) + C * r**D
    
    
    
# Initial guess for parameters
initial_guess = [1, 1 ,0,0.3,0.25]

# Fit the model to the data
params, covariance = curve_fit(model_func, (x, y), z, initial_guess)

# Extracting the parameters

print("Fitted parameters:", params)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the original data
ax.scatter(x, y, z, color='r', marker='o', label='Data')

# Creating a meshgrid for the plane representing the fitted 2D function
X, Y = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
Z_fit = model_func_d(X, Y, params[0], params[1], params[2], params[3], params[4])

# Plotting the fitted plane
ax.plot_surface(X, Y, Z_fit, color='b', alpha=0.5, label='Fitted Plane')

ax.set_xlabel('Y')
ax.set_ylabel('r[GeV^-1]')
ax.set_zlabel('N')

plt.title('Fitted 2D Function in 3D Space')


plt.title('Fitted 2D Function in 3D Space')
plt.savefig('tesddt.png', dpi=120)

"""
