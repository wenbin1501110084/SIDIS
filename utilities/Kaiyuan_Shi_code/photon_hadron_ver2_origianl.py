############################################################
#                                                          #
#             Isolated Photon-Hadron Production            #
#      in proton-proton and proton-nucleus collisions      #
#                    by Kaiyuan Shi                        #
#                                                          #
############################################################


import numpy as np
from numpy import cos, exp, sqrt, log, log10  # Import functions
from numpy import e, pi, euler_gamma, inf  # Import constants
import matplotlib.pyplot as plt

from scipy import special
from scipy.integrate import dblquad, quad, nquad, simps, trapz 
from scipy.special import jv, kv, j0
from scipy.interpolate import interp2d, interp1d
from FBT import FBT

from pandas import read_table

from PDFFF import xgp
from PDFFF import xup, xdp, xsp, xgp, xuA, xdA, xsA, xgA, Du, Dd, Ds # FF and PDF

from copy import copy  # Remember: arrayA=arrayB would result in simultaneous changes, so we need it.

from datetime import datetime  # Used for progress tracking.
from IPython.display import clear_output

import numba as nb  # Faster Calculation
from numba import jit
import multiprocessing as mp  # Multiprocessing on multiple cores.
fbt = FBT(0)

############################## Argument inputs ##############################
from argparse import ArgumentParser
parser = ArgumentParser(description = 'Make graphs for photon-hadron production.')
########## Basic inputs ##########
parser.add_argument('--s', type=float, default=8162.
                    , help='Sqrt of CoM energy, default by 8162 for LHCb.')
parser.add_argument('--A', type=float, default=1.
                    , help='Atomic number, default by 1 for proton.')
parser.add_argument('--N', type=float, default=1.
                    , help='Ratio between nuclear and proton saturation scale, default by 1 for proton.')

parser.add_argument('--option', type=str, choices=['CS', 'ALL', 'DM'], default='CS'
                    , help='Command to perform. Options: MV, MVe, MVg.')

parser.add_argument('--P1s', nargs='+', type = float, default = [1., 5., 10., 15.]
                    , help = 'Ranges of P1.')
parser.add_argument('--P2s', nargs='+', type = float, default = [0.3, 2., 5., 10.]
                    , help = 'Ranges of P2.')
parser.add_argument('--ymin', type = float, default = 2.0
                    , help = 'Maximum rapidity.')
parser.add_argument('--ymax', type = float, default = 4.5
                    , help = 'Minimum rapidity.')
parser.add_argument('--pstep', type = int, default = 5
                    , help = 'Steps in momentum when integrating.')
parser.add_argument('--ystep', type = int, default = 3
                    , help = 'Steps in rapidity when integrating.')
parser.add_argument('--phistep', type = int, default = 60
                    , help = 'Steps in angular correlation.')


parser.add_argument('--data', type = int, default = 0
                    , help = 'Whether to include data when producing the graphs.')
parser.add_argument('--noneg', type = int, default = 0
                    , help = 'Whether to include data when producing the graphs.')
parser.add_argument('--abs', type = int, default = 0
                    , help = 'Whether to include data when producing the graphs.')

parser.add_argument('--core', type = int, default = 43
                    , help = 'Number of cores to use.')

parser.add_argument('--suffix', type = str, default = 'TEST'
                    , help = 'Suffix for files generated/loaded.')
parser.add_argument('--yub', type = float, default= 2.6
                    , help = 'The upper bound of y to be used in plots.')
parser.add_argument('--xlb', type = float, default= 0
                    , help = 'The upper bound of x to be used in plots.')

########## Extra inputs if option == 'CS' ##########
parser.add_argument('--model', type=str, choices=['MV', 'MVe', 'MVg'], default='MVe'
                    , help='Name of the dipole model to use, MVe by default to avoid negative values. Options: MV, MVe, MVg.')
parser.add_argument('--NPcoeff', type=float, default=1.
                    , help='Coefficient before the non-perturbative Sudakov factor for uncertainty calculation.')

########## Extra inputs if option == 'ALL' ##########
parser.add_argument('--norm', type = int, default = 1
                    , help = 'Whether to normalize the curve to unity.')
parser.add_argument('--ranged', type = int, default = 0
                    , help = 'Whether to limit the normalization to the range we are looking at.')
parser.add_argument('--shifted', type = int, default = 1
                    , help = 'Whether to shift the data points by pedestal, if data==True.')

########## Extra inputs if option == 'DM' ##########
parser.add_argument('--DMsuffix', nargs='+', type = str, default= []
                    , help = 'The first suffix(group) used to generate the graph.')
parser.add_argument('--type', type = str, choices=['C', 'S', 'CS', 'A'], default= 'S'
                    , help = 'The type of curve to draw. Options: C(CGC), S(Sudakov), CS(CGC+Sudakov), A(All)')
parser.add_argument('--relative', type = int, default= 0
                    , help = 'Whether to make relative curves if normalize is given True.')
parser.add_argument('--ratio', type = int, default= 0
                    , help = 'Whether to find the ratio between curves.')
parser.add_argument('--band', type = int, default= 0
                    , help = 'Whether to draw the error band.')
parser.add_argument('--replace', nargs='+', type = str, default= ['noneg']
                    , help = 'The str the remove in the graphs.')

args = parser.parse_args()

# Calculation parameters
s = (args.s)**2.
A = args.A  # The atomic number.
RA = A**(1./3.)  # Target radius.
N = args.N  # Ratio between nuclear and proton saturation scale
model_type = args.model  # Model name
NPcoeff = args.NPcoeff  # The coefficient before the NP-Sudakov for the uncertainty related.

# Kinematics
P1s = args.P1s
P2s = args.P2s
ymin = args.ymin
ymax = args.ymax
psteps = args.pstep
ysteps = args.ystep
phisteps = args.phistep
Y1 = np.linspace(ymin, ymax, ysteps, endpoint = False)
Y2 = np.linspace(ymin, ymax, ysteps, endpoint = False)
PHI = np.linspace(0, pi, phisteps) # The azimuthal angles where we want to calculate the cross sections.

# Extra parameters.
has_data = args.data
each_noneg = args.noneg
each_abs = args.abs
suffix = str(args.suffix)
yub = args.yub
xlb = args.xlb
is_norm = args.norm
is_ranged = args.ranged
is_shifted = args.shifted
DMsuffix = args.DMsuffix
curve_type = args.type
is_relative = args.relative
is_ratio = args.ratio
has_band = args.band
text_replace = args.replace

#################### Paths ####################
CS_path = './CS/'  # Path for output cross section tables.
fig_path = '../fig_output_FBT/'  # Path for output figures.
data_path = './PHENIX_pp200/'  # Path for input data points if available.
allfig_path = '../figout_allCS/'
DMfig_path = '../figout_differentModels/'
#################### Constants Used in Calculation ####################
Nc = 3.  # Number of colors.
CA = Nc
CF = (Nc**2. -1.) / (2. * Nc)
Nf = 3.  # Number of flavors, currently 3: u, d, s.
alphae = 1./137.  # EM constant.
R = 0.3 # The cone cut


#################### Splitting Functions ####################
Pgq = lambda z: CF * (1. + (1. - z)**2.) / z  # gluon to quark
Pqg = lambda z: CF * (z**2. + (1. - z)**2.) / 2.  # quark to gluon
Pgg = lambda z: 2. * Nc * (z/(1.-z) + (1.-z)/z + z*(1.-z))  # gluon to gluon
Pqgamma = lambda z: (1. + (1.-z)**2.) / z  # quark to photon


############################## Kinematic Variables ##############################
x_mode = 2  # The mode we are calculating xp and xA. 1 for exact, 2 for approximate.
if x_mode == 1:
    xp = lambda k1, k2, y1, y2: s**(-1./2.) * (k1 * exp(y1) + k2 * exp(y2))  # Feynman-x variable in the + direction.
    xA = lambda k1, k2, y1, y2: s**(-1./2.) * (k1 * exp(-y1) + k2 * exp(-y2))  # Feynman-x variable in the - direction.
else:
    xp = lambda k1, k2, y1, y2: s**(-1./2.) * max(k1, k2) * (exp(y1) + exp(y2))
    xA = lambda k1, k2, y1, y2: s**(-1./2.) * max(k1, k2) * (exp(-y1) + exp(-y2))
x1 = xp
x2 = xA

Q2 = lambda k1, k2, y1, y2: xp(k1,k2,y1,y2) * xA(k1,k2,y1,y2) * s  # Hard scale.
z = lambda k1, k2, y1, y2: k1*exp(y1)/(k1*exp(y1) + k2*exp(y2))  # Longitudinal momentum fraction of the first parton.
kt2 = lambda k1, k2, phi: abs(k1**2. + k2**2. + 2.*k1*k2*cos(phi))  # Imbalance transverse momenta squared.
kt = lambda k1, k2, phi: sqrt(abs(k1**2. + k2**2. + 2.*k1*k2*cos(phi)))  # Imbalance transverse momenta.

############################## PDF and FF ##############################


############################## Dipole Models ##############################
Q0 = 2.0494 # The Q scale at which we are matching the curves
x0 = 0.01
Lam = 0.241  # QCD scale Lambda_QCD.

Qs02_ = N * 0.104  # Saturation scale for MV model.
Qs02_e = N * 0.060  # Saturation scale for MVe model.
ec = 18.9
Qs02_g = N * 0.169  # Saturation scale for MVgamma model.
gamma = 1.119

Qs2 = lambda x, Qs02: Qs02 * (x0/x)**0.3  # Simple evolution of the saturational scale in x.


def MV(r, x):  # MV model
    if x >= 1.:
        result = 0.

    elif x >= x0:
        result = exp(-r**2.*Qs02_/4. * log(1./(Lam*r) + e))
        result *= xgp(x, Q0)

    else:
        result = exp(-r**2. * Qs2(x, Qs02_)/4. * log(1./(Lam*r) + e))
    return(result)   

def MVg(r, x):  # MV_gamma model
    if x >= 1:
        result = 0

    elif x >= x0:
        result = exp(-(r**2.*Qs02_g)**gamma/4. * log(1./(Lam*r) + e))
        result *= xgp(x, Q0)

    else:
        result = exp(-(r**2. * Qs2(x, Qs02_g))**gamma/4. * log(1./(Lam*r) + e))
    return(result)

def MVe(r, x):  # MV_e model
    if x >= 1:
            result = 0

    elif x >= x0:
        result = exp(-r**2.*Qs02_e/4. * log(1./(Lam*r) + ec*e))
        result *= xgp(x, Q0)

    else:
        result = exp(-r**2. * Qs2(x, Qs02_e)/4. * log(1./(Lam*r) + ec*e))

#################### radial Laplacian of Dipole Models ####################
def rlaplaceN(r, x):  # MV model
    if x >= 1 or r == 0:
        result = 0.

    elif x < x0:
        Q = sqrt(Qs2(x, Qs02_))
        result = -(Q**2. * (1./(Lam*r) + e)**(-1./4.*Q**2. * r**2.)*(-4.*(4.*e**2. * Lam**2. * r**2.\
            + e*Lam*r*(Q**2. * r**2. + 8.) + Q**2. * r**2. + 4.)*log(1./(Lam*r)\
            + e) + 4.*Q**2. * r**2.*(e*Lam*r + 1. )**2. * log(1./(Lam*r) + e)**2. + 12.*e*Lam*r\
            + Q**2. * r**2. + 16.))/(16.*(e*Lam*r + 1.)**2.)

    else:
        Q = sqrt(Qs02_)
        result = -(Q**2. * (1./(Lam*r) + e)**(-1./4.*Q**2. * r**2.)*(-4.*(4.*e**2. * Lam**2. * r**2.\
            + e*Lam*r*(Q**2. * r**2. + 8.) + Q**2. * r**2. + 4.)*log(1./(Lam*r)\
            + e) + 4.*Q**2. * r**2.*(e*Lam*r + 1.)**2. * log(1./(Lam*r) + e)**2. + 12.*e*Lam*r\
            + Q**2. * r**2. + 16.))/(16.*(e*Lam*r + 1.)**2.)
        # result *= ((1-x)/(1-x0))**4 * (x0/x)**(0.15)
        result *= xgp(x, Q0)

    return(result)

def rlaplaceNe(r, x):  # MV_e model
    if x >= 1 or r == 0:
        result = 0.

    elif x < x0:
        Q = sqrt(Qs2(x, Qs02_e))
        result = -(Q**2. * (1./(Lam*r) + ec*e)**(-1./4.*Q**2. * r**2.)*(-4.*(4.*(ec*e)**2. * Lam**2. * r**2.\
            + ec*e*Lam*r*(Q**2. * r**2. + 8.) + Q**2. * r**2. + 4.)*log(1./(Lam*r)\
            + ec*e) + 4.*Q**2. * r**2.*(ec*e*Lam*r + 1.)**2. * log(1./(Lam*r) + ec*e)**2. + 12.*ec*e*Lam*r\
            + Q**2. * r**2. + 16.))/(16.*(ec*e*Lam*r + 1.)**2.)

    else:
        Q = sqrt(Qs02_e)
        result = -(Q**2. * (1./(Lam*r) + ec*e)**(-1./4.*Q**2. * r**2.)*(-4.*(4.*(ec*e)**2. * Lam**2. * r**2.\
            + ec*e*Lam*r*(Q**2. * r**2. + 8.) + Q**2. * r**2. + 4.)*log(1./(Lam*r)\
            + ec*e) + 4.*Q**2. * r**2.*(ec*e*Lam*r + 1.)**2. * log(1./(Lam*r) + ec*e)**2. + 12.*ec*e*Lam*r\
            + Q**2. * r**2. + 16.))/(16.*(ec*e*Lam*r + 1.)**2.)
        # result *= ((1-x)/(1-x0))**4 * (x0/x)**(0.15)
        result *= xgp(x, Q0)

    return(result)

def rlaplaceNg(r, x):  # MV_gamma model
    if x >= 1 or r == 0:
        result = 0.

    elif x < x0:
        Q = sqrt(Qs2(x, Qs02_g)) 
        result = -((Q*r)**(2.*gamma) * (1./(Lam*r) + e)**(-1./4. * (Q*r)**(2.*gamma))\
            * (4.*gamma * (e*Lam*r + 1.) * log(1./(Lam*r) + e)\
            * ((Q*r)**(2.*gamma) * ((e*gamma*Lam*r + gamma)\
            * log(1./(Lam*r) + e) - 1.) - 2.*(1. + 2.*gamma)\
            * (e*Lam*r + 1.)) + (Q*r)**(2.*gamma) + 16.*gamma\
            * (e*Lam*r + 1.) + 4.)) / (16.*r**2. * (e*Lam*r + 1.)**2.)

    else:
        Q = sqrt(Qs02_g)
        result = -((Q*r)**(2.*gamma) * (1./(Lam*r) + e)**(-1./4. * (Q*r)**(2.*gamma))\
            * (4.*gamma * (e*Lam*r + 1.) * log(1./(Lam*r) + e)\
            * ((Q*r)**(2.*gamma) * ((e*gamma*Lam*r + gamma)\
            * log(1./(Lam*r) + e) - 1.) - 2.*(1. + 2.*gamma)\
            * (e*Lam*r + 1.)) + (Q*r)**(2.*gamma) + 16.*gamma\
            * (e*Lam*r + 1.) + 4.)) / (16.*r**2. * (e*Lam*r + 1.)**2.)

        # result *= ((1-x)/(1-x0))**4 * (x0/x)**(0.15)
        result *= xgp(x, Q0)            
    return(result)

if model_type == 'MV':
    S = MV
    rLaplacian = rlaplaceN
elif model_type == 'MVg':
    S = MVg
    rLaplacian = rlaplaceNg
else:
    S = MVe
    rLaplacian = rlaplaceNe

############################## Strong Coupling and Sudakov Calculation ##############################
bmax = 1.5  # For the Sudakov calculation.

beta0_p = 0.623  # for proton alpha_s fit
Lam_p = 0.186  # for proton alpha_s fit
beta0_A = 0.728  # for nuclear alpha_s fit
Lam_A = 0.227  # for nuclear alpha_s fit
beta = (11. - 2.*Nf/3.) / 12.

alphap = lambda mu2: 1. / (beta0_p * log(mu2 / Lam_p**2.))  # Strong coupling fit from the proton PDF. 
alphaA = lambda mu2: 1. / (beta0_A * log(mu2 / Lam_A**2.))  # Strong coupling fit from the nuclear PDF.

@jit(nopython=True)
def bstar2(b):  # b*.
    if b >= 1e9: # There would be no change in b* beyond b=1e9.
        result = bmax**2
    else:
        result = b**2 / (1. + b**2/bmax**2)
    return(result)

def mub(b):  # Factorization scale according to b*.
    if b == 0: # To avoid the error caused by division by zero.
        result = float(inf)
    else:
        result = 2. * exp(-euler_gamma) / sqrt(bstar2(b))
    return(result)

def Sq_nonpert(b, Q):  # NP Sudakov factor caused by quarks.
    Q02 = 2.4
    g1 = 0.212
    g2 = 0.84
    if b == 0 or b == np.inf or NPcoeff == 0:
        result = 0
    else:
        result = g1/2. * b**2. + 1./4. * g2/2. * log(Q**2 / Q02) * log(b**2 / bstar2(b))
    return(result)
Sg_nonpert = lambda b, Q: CA/CF * Sq_nonpert(b, Q)  # NP Sudakov factor caused by gluons.

def Sudakov(b, Q, is_sud = True):  # Total Sudakov factor
    beta0 = beta0_p
    Lam = Lam_p
    mu = mub(b)

    if mu**2 > Q**2:
        Ssud = 0
        Snonpert = 0
    else:
        if is_sud:
            A = 1. / pi * (CF + CA/2.)  
            B = -1. / pi * (3./2. * CF + CA * beta)
            Snonpert = 2. * Sq_nonpert(b, Q) + Sg_nonpert(b, Q)
        else:
            A = 1. / pi * (CF + CA/2.)
            B = -1. / pi * (3./2. * CF)
            Snonpert = 2. * Sq_nonpert(b,Q)
        Ssud = -2.*A*log(Q/mu) + (B+2*A*log(Q/Lam)) * log((log(Q/Lam)) / log(mu/Lam))
        Ssud = Ssud.real / beta0

    result = exp(- Ssud - NPcoeff*Snonpert)
    return(result)

def hard_factor(k1, k2, y1, y2, phi, is_sud = True):  # Hard factor.
    Z = z(k1, k2, y1, y2)
    if is_sud:
        qkg = k1**2. / (Z * (1.- Z)) / 2.  # q dot k_gamma.
    else:
        qkg = ((1.-Z)/Z * k1**2. + Z/(1.-Z) * k2**2. - 2.*k1*k2*cos(phi)) / 2.
    result = alphae / 2. / Nc * Pqgamma(Z) / qkg * Z**2. / k1**2.
    return(result)


#################### Functions to Modify Arrays ####################
def ntu(arr, angle):  # Normalize arrays to unity.
    arr_new = copy(arr)

    for i in range(len(arr)):
        arr_new[i] = max(0, arr_new[i])

    coeff = 1. / simps(arr_new, angle)
    result = copy(arr) * coeff
    
    return(result, coeff)

def symmetrize(arr):  # Symmetrize an array with domain from 0 to pi and expand it as from 0 to 2pi.
    arr_reverse = arr[::-1]
    result = np.hstack((arr, arr_reverse))
    return(result)

def non_negative(arr):  # Replace negative elements in the array by 0 and return the array.
    arr_new = copy(arr)
    for i in range(len(arr_new)):
        arr_new[i] = max(arr_new[i], 0)
    return(arr_new)

def get_shift(data):  # Get the value to shift the experimental data by using the off-peak cross sections.
    count = 0
    off_peak = []
    angle = data[:, 0]
    val = data[:, 1]
    
    for i in range(len(data)):
        if (angle[i] <= pi - 1.1) or (angle[i] >= pi + 1.1):
            if val[i] >= 0:
                off_peak.append(val[i])
                count += 1
            
    if len(off_peak) >= 4:
       off_peak[off_peak.index(min(off_peak))] = 0
       count -= 1
       off_peak[off_peak.index(max(off_peak))] = 0
       count -= 1
    
    if count <= 2:
        shift = 0
    else:
        shift = sum(off_peak) / count
        
    return(shift)

############################## Calculate the Differential Cross Sections ##############################
def CGCb(k1, k2, y1, y2, z, phi):  # b-integrand for CGC cross section.
    x_p = xp(k1, k2, y1, y2)
    x_A = xA(k1, k2, y1, y2)
    k = kt(k1, k2, phi)
    mu = max(k1, k2)

    if x_p >= 1 or x_A >= 1:
        result = 0.

    else:
        if k <= 1:
            funcCGC = lambda b: b * j0(k * b) * rLaplacian(b, x_A)
            result = quad(funcCGC, 0, np.inf, limit = 1000)[0]
        else:
            funcCGC = lambda b: b * rLaplacian(b, x_A)
            funcCGC = np.vectorize(funcCGC)
            result = fbt.fbt(funcCGC, k, N = 100, Q = 1., option = 2) * 2. * pi

        result *= ((Du(z, mu) * xup(x_p, mu) * (2./3.)**2.\
            + Dd(z, mu) * xdp(x_p, mu) * (-1./3.)**2.\
            + Ds(z, mu) * xsp(x_p, mu) * (-1./3.)**2.) / z**2.)\
            * hard_factor(k1, k2, y1, y2, phi, is_sud = True)
    return(result)

def DCS_CGC(k1, k2, y1, y2, phi):  # Differential cross section for CGC-only.

    if ((y1-y2)**2 + phi**2 <= R**2):
        result = 0.

    else:
        func_CGC = lambda zh: CGCb(k1, k2/zh, y1, y2, zh, phi)
        result = quad(func_CGC, 1e-8, 1, limit = 1000)[0]

        if each_noneg == True:
            result = max(result, 0)
        elif each_abs == True:
            result = abs(result)

    return(result * k1 * k2)

def SudbA(k1, k2, y1, y2, z, phi):  # Nuclear GDF part of b-integrand for Sudakov-only cross section.
    x_p = xp(k1, k2, y1, y2)
    x_A = xA(k1, k2, y1, y2)
    k = kt(k1, k2, phi)
    Q = sqrt(Q2(k1, k2, y1, y2))

    if x_p >= 1 or x_A >= 1:
        result = 0.
        
    else:
        funcsud = lambda b: b * alphap(mub(b)**2.) * Sudakov(b, Q, is_sud = True)\
            * xgA(x_A, mub(b)) * (Du(z, mub(b)) * xup(x_p, mub(b)) * (2./3.)**2.\
            + Dd(z, mub(b)) * xdp(x_p, mub(b)) * (-1./3.)**2.\
            + Ds(z, mub(b)) * xsp(x_p, mub(b)) * (-1./3.)**2.) / z**2.\
            * hard_factor(k1, k2, y1, y2, phi, is_sud = True)
        funcsud = np.vectorize(funcsud)

        if k <= 1:
            # funcsud = lambda b: b * j0(k * b) * xgp(x_A, mub(b)) * alpha(mub(b)**2) * Sud_sud(b, Q)\
            #     * (Du(z, mub(b)) * xup(x_p, mub(b)) * (2./3.)**2\
            #     + Dd(z, mub(b)) * xdp(x_p, mub(b)) * (-1./3.)**2\
            #     + Ds(z, mub(b)) * xsp(x_p, mub(b)) * (-1./3.)**2) / z**2.
            # result = quad(funcsud, 0, np.inf, limit = 1000)[0]
            result = fbt.fbt(funcsud, 1., N = 100, Q = 1., option = 2) * 2. * pi

        else:
            result = fbt.fbt(funcsud, k, N = 100, Q = 1., option = 2) * 2. * pi

    return(result)

def Sudbp(k1, k2, y1, y2, z, phi):  # Proton GDF part of b-integrand for Sudakov-only cross section.
    x_p = xp(k1, k2, y1, y2)
    x_A = xA(k1, k2, y1, y2)
    k = kt(k1, k2, phi)
    Q = sqrt(Q2(k1, k2, y1, y2))

    if x_p >= 1 or x_A >= 1:
        result = 0.
        
    else:
        funcsud = lambda b: b * alphap(mub(b)**2.) * Sudakov(b, Q, is_sud = True)\
            * xgp(x_p, mub(b)) * (Du(z, mub(b)) * xuA(x_A, mub(b)) * (2./3.)**2.\
            + Dd(z, mub(b)) * xdA(x_A, mub(b)) * (-1./3.)**2.\
            + Ds(z, mub(b)) * xsA(x_A, mub(b)) * (-1./3.)**2.) / z**2.\
            * hard_factor(k2, k1, y2, y1, phi, is_sud = True)
            
        funcsud = lambda b: b * alphap(mub(b)**2.) * Sudakov(b, Q, is_sud = True)\
            * xgA(x_A, mub(b)) * (Du(z, mub(b)) * xup(x_p, mub(b)) * (2./3.)**2.\
            + Dd(z, mub(b)) * xdp(x_p, mub(b)) * (-1./3.)**2.\
            + Ds(z, mub(b)) * xsp(x_p, mub(b)) * (-1./3.)**2.) / z**2.\
            * hard_factor(k1, k2, y1, y2, phi, is_sud = True)

        funcsud = np.vectorize(funcsud)

        if k <= 1:
            result = fbt.fbt(funcsud, 1., N = 100, Q = 1., option = 2) * 2. * pi

        else:
            result = fbt.fbt(funcsud, k, N = 100, Q = 1., option = 2) * 2. * pi

    return(result)

def DCS_Sud(k1, k2, y1, y2, phi):  # Differential cross section for Sudakov-only.

    if ((y1-y2)**2 + phi**2 <= R**2):
        result = 0.

    else:
        func_sud = lambda zh: SudbA(k1, k2/zh, y1, y2, zh, phi) + Sudbp(k1, k2/zh, y1, y2, zh, phi)
        result = quad(func_sud, 1e-8, 1, limit = 1000)[0]

        if each_noneg == True:
            result = max(result, 0)
        elif each_abs == True:
            result = abs(result)
            
    return(result * k1 * k2)

def CGCSudbA(k1, k2, y1, y2, z, phi):  # Nuclear gluon dipole part of b-integrand for CGC+Sudakov cross section.
    x_p = xp(k1, k2, y1, y2)
    x_A = xA(k1, k2, y1, y2)
    k = kt(k1, k2, phi)
    Q = sqrt(Q2(k1, k2, y1, y2))

    if x_p >= 1 or x_A >= 1:
        result = 0.
        
    else:
        funcsud = lambda b: b * Sudakov(b, Q, is_sud = False)\
            * rLaplacian(b, x_A) * (Du(z, mub(b)) * xup(x_p, mub(b)) * (2./3.)**2.\
            + Dd(z, mub(b)) * xdp(x_p, mub(b)) * (-1./3.)**2.\
            + Ds(z, mub(b)) * xsp(x_p, mub(b)) * (-1./3.)**2.) / z**2.\
            * hard_factor(k1, k2, y1, y2, phi, is_sud = True)
        funcsud = np.vectorize(funcsud)

        if k <= 1:
            # funcsud = lambda b: b * j0(k * b) * xgp(x_A, mub(b)) * alpha(mub(b)**2) * Sud_sud(b, Q)\
            #     * (Du(z, mub(b)) * xup(x_p, mub(b)) * (2./3.)**2\
            #     + Dd(z, mub(b)) * xdp(x_p, mub(b)) * (-1./3.)**2\
            #     + Ds(z, mub(b)) * xsp(x_p, mub(b)) * (-1./3.)**2) / z**2.
            # result = quad(funcsud, 0, np.inf, limit = 1000)[0]
            result = fbt.fbt(funcsud, 1., N = 100, Q = 1., option = 2) * 2. * pi

        else:
            result = fbt.fbt(funcsud, k, N = 100, Q = 1., option = 2) * 2. * pi

    return(result)

def CGCSudbp(k1, k2, y1, y2, z, phi):  # Proton gluon dipole part of b-integrand for CGC+Sudakov cross section.
    x_p = xp(k1, k2, y1, y2)
    x_A = xA(k1, k2, y1, y2)
    k = kt(k1, k2, phi)
    Q = sqrt(Q2(k1, k2, y1, y2))

    if x_p >= 1 or x_A >= 1:
        result = 0.
        
    else:
        funcsud = lambda b: b * Sudakov(b, Q, is_sud = False)\
            * rLaplacian(b, x_p) * (Du(z, mub(b)) * xuA(x_A, mub(b)) * (2./3.)**2.\
            + Dd(z, mub(b)) * xdA(x_A, mub(b)) * (-1./3.)**2.\
            + Ds(z, mub(b)) * xsA(x_A, mub(b)) * (-1./3.)**2.) / z**2.\
            * hard_factor(k2, k1, y2, y1, phi, is_sud = True)
        funcsud = np.vectorize(funcsud)

        if k <= 1:
            result = fbt.fbt(funcsud, 1., N = 100, Q = 1., option = 2) * 2. * pi

        else:
            result = fbt.fbt(funcsud, k, N = 100, Q = 1., option = 2) * 2. * pi

    return(result)

def DCS_CGCSud(k1, k2, y1, y2, phi):  # Differential cross section for CGC+Sudakov.

    if ((y1-y2)**2 + phi**2 <= R**2):
        result = 0.

    else:
        func_sud = lambda zh: CGCSudbA(k1, k2/zh, y1, y2, zh, phi) + CGCSudbp(k1, k2/zh, y1, y2, zh, phi)
        result = quad(func_sud, 1e-8, 1, limit = 1000)[0]

        if each_noneg == True:
            result = max(result, 0)
        elif each_abs == True:
            result = abs(result)
            
    return(result * k1 * k2)

######################################## Make ########################################
######################################## Plot ########################################
def single_CS(p1_min, p1_max, p2_min, p2_max):  # Calculate total cross sections for a certain range of momenta.
    _P1 = np.linspace(p1_min, p1_max, psteps, endpoint = False)
    _P2 = np.linspace(p2_min, p2_max, psteps, endpoint = False)
    dp1 = abs(_P1[1] - _P1[0])
    dp2 = abs(_P2[1] - _P2[0])
    dy1 = abs(Y1[1] - Y1[0])
    dy2 = abs(Y2[1] - Y2[0])

    CSname = CS_path + suffix + '_' + str(p1_min) + '_' + str(p1_max) + '_' + str(p2_min) + '_' + str(p2_max) + '.txt'
    figname = fig_path + suffix + '_' + str(p1_min) + '_' + str(p1_max) + '_' + str(p2_min) + '_' + str(p2_max) + '.png'
    start = datetime.now()
    print(CSname + " STARTING AT " + start.strftime("%Y/%m/%d %H:%M:%S") + ', AND', end = '')

    if has_data == True:  # Get data points if needed.
        dataname = data_path + str(p1_min) + '_' + str(p1_max) + '_' + str(p2_min) + '_' + str(p2_max) + '.txt'
        datapoints = np.array(read_table(dataname, sep = '&'))

    _PHI = copy(PHI)  # The angular correlations for calculation.
    size = int(len(_PHI))  # The size of tables to be generated (number of rows).
    CGC = np.zeros(size)  # Initialize array for CGC.
    Sud = np.zeros(size)  # Initialize array for Sudakov.
    CGCSud = np.zeros(size)  # Initialize array for CGC+Sudakov.
    steps_total = psteps**2 * ysteps**2

    # Create the working pool for multiprocessing.
    pool = mp.Pool(args.core)
    jobs_CGC = []
    jobs_Sud = []
    jobs_CGCSud = []
    for i in range(size):
        jobs_CGC.append([])
        jobs_Sud.append([])
        jobs_CGCSud.append([])

    for p1 in _P1:
        for p2 in _P2:
            for y1 in Y1:
                for y2 in Y2:
                    for i in range(size):
                        phi = _PHI[i]        
                        jobs_CGC[i].append(pool.apply_async(DCS_CGC, args = (p1, p2, y1, y2, phi)))
                        jobs_Sud[i].append(pool.apply_async(DCS_Sud, args = (p1, p2, y1, y2, phi)))
                        jobs_CGCSud[i].append(pool.apply_async(DCS_CGCSud, args = (p1, p2, y1, y2, phi)))

    for i in range(size):
        for j in range(steps_total):

            CGC[i] += (jobs_CGC[i][j]).get()
            Sud[i] += (jobs_Sud[i][j]).get()
            CGCSud[i] += (jobs_CGCSud[i][j]).get()

    CGC *= dp1 * dp2 * dy1 * dy2 * RA**2
    Sud *= dp1 * dp2 * dy1 * dy2 * RA**2
    CGCSud *= dp1 * dp2 * dy1 * dy2 * RA**2

    # Symmetrize to the entire 0 to 2pi region.
    dPHI = np.linspace(0, 2*pi, 2*size) 
    CGC = symmetrize(CGC)
    Sud = symmetrize(Sud)
    CGCSud = symmetrize(CGCSud)

    # Plot the curves.
    CGCplot = ntu(CGC, dPHI)[0]
    Sudplot = ntu(Sud, dPHI)[0]
    CGCSudplot = ntu(CGCSud, dPHI)[0]
    plt.figure(figsize = (6.5, 6))
    if has_data == True:
        plt.errorbar(datapoints[:, 0]
            , ntu(datapoints[:, 1] - get_shift(datapoints), datapoints[:, 0])[0]
            , yerr = ntu(datapoints[:, 1] - get_shift(datapoints), datapoints[:, 0])[1] * datapoints[:, 2]
            , fmt = 'o', capsize = 3, markersize=6, label = 'PHENIX')
        
    plt.plot(dPHI, CGCplot, color = 'orange', label = 'CGC')
    plt.plot(dPHI, Sudplot, color = 'teal', linewidth = 1.5, ls = '--',label = 'sud')
    plt.plot(dPHI, CGCSudplot, color = 'magenta', linewidth = 3, label = 'CGC + sud', alpha = 0.7)
    
    plt.title('Cross Section for $p_1 \in $[' + str(p1_min) + ',' + str(p1_max) + '], ' +
                                '$p_2 \in $[' + str(p2_min) + ',' + str(p2_max) + '], ' +
                                '$y \in $[' + str(ymin) + ',' + str(ymax) + ']')
    plt.xlabel('$\Delta \phi$', size = 12)
    plt.ylabel('$\\frac{1}{\sigma} \\frac{d \sigma}{d \Delta \phi}$', size = 12)
    plt.legend()

    plt.savefig(figname, bbox_inches='tight')

    # Save the Cross Sections.
    CS_table = np.zeros((len(CGC), 3))

    for i in range(len(CGC)):
        CS_table[i] = [CGC[i], Sud[i], CGCSud[i]]

    np.savetxt(CSname, CS_table, delimiter=' & ', header = 'CGC & Sud & CGCSud', newline='\n', fmt='%.16f')
    end = datetime.now()
    print(' ENDING AT ' + end.strftime("%Y/%m/%d %H:%M:%S") + ', TIME TAKEN: ' + str(end-start).split('.')[0])

    return

def getCS():  # Get the cross section tables for the momenta bins.
    for p1_ind in range(len(P1s)-1):
        for p2_ind in range(len(P2s)-1):
            p1min = P1s[p1_ind]
            p1max = P1s[p1_ind+1]
            p2min = P2s[p2_ind]
            p2max = P2s[p2_ind+1]
            single_CS(p1min, p1max, p2min, p2max)
    return

def drawALL():  # Get the cross sections with different momenta bins but same kinematics.
    dPHI = np.linspace(0, 2.*pi, 2 * len(PHI))
    xub = 2*pi-xlb  # x upper bound, x lower bound has been defined in the argument.
    ylb = -0.1  # y lower bound, y upper bound has been defined in the arguments.

    ind_ignore = int(xlb / (2*pi) * len(dPHI))  # Index in list to ignore when ranged is True.

    # Get the name of the figure.
    sufraw = suffix  # Raw suffix to extract the tables later. 
    figname = allfig_path + 'ALL_' + suffix + '.png'
    if is_norm:
        figname = figname.replace('.png', '_norm.png')
    if is_ranged:
        dPHI = dPHI[ind_ignore : 2*len(PHI) - ind_ignore]
        figname = figname.replace('.png', ('_ranged' + str(round(xlb, 3)) + '.png'))
    if is_shifted:
        figname = figname.replace('.png', '_shifted.png')
    if has_data:
        figname = figname.replace('.png', '_withdata.png')


    # Draw the figure.
    fig = plt.figure(figsize = (16, 16))
    gs = fig.add_gridspec(len(P2s)-1, len(P1s)-1, hspace=0, wspace=0)
    axsp = gs.subplots(sharex=True, sharey=True)

    index_row = 0  # Row index of the graph.
    index_column = 0  # Column index of the graph.

    for p1_ind in range(len(P1s)-1):
        for p2_ind in range(len(P2s)-1):
            p1min = P1s[p1_ind]
            p1max = P1s[p1_ind+1]
            p2min = P2s[p2_ind]
            p2max = P2s[p2_ind+1]

            # Get the cross sections from the tables.
            CS = np.array(read_table(CS_path + sufraw + '_'\
                + str(p1min) + '_' + str(p1max) + '_'\
                + str(p2min) + '_' + str(p2max) + '.txt', sep = '&'))
            CGC = CS[:, 0]
            Sud = CS[:, 1]
            CGCSud = CS[:, 2]
            
            if has_data:  # Get the data points if needed.
                datapoints = np.array(read_table(data_path\
                    + str(p1min) + '_' + str(p1max) + '_'\
                    + str(p2min) + '_' + str(p2max) + '.txt', sep = '&'))
                
            if is_ranged:  # Remove cross sections outside the range we are looking at if is_ranged==True.
                CGC = CGC[ind_ignore : 2*len(PHI) - ind_ignore]
                Sud = Sud[ind_ignore : 2*len(PHI) - ind_ignore]
                CGCSud = CGCSud[ind_ignore : 2*len(PHI) - ind_ignore]

            if is_norm:  # Normalize the curves to unity if needed.
                CGC = ntu(CGC, dPHI)[0]
                Sud = ntu(Sud, dPHI)[0]
                CGCSud = ntu(CGCSud, dPHI)[0]
            
            plt.xlim(xlb, xub)
            plt.ylim(ylb, yub)
            axsp[index_row, index_column].plot(dPHI, CGC, color = 'orange', label = 'CGC')
            axsp[index_row, index_column].plot(dPHI, Sud, color = 'teal', linewidth = 3, ls = ':',label = 'sud')
            axsp[index_row, index_column].plot(dPHI, CGCSud, color = 'magenta', linewidth = 3, label = 'CGC + sud', alpha = 0.7)
            axsp[index_row, index_column].text((xlb+(xub-xlb)*0.75), yub*0.9, (str(p1min) + "$< k_{γt} <$" + str(p1max)))
            axsp[index_row, index_column].text((xlb+(xub-xlb)*0.75), yub*0.8, (str(p2min) + "$< P_{ht} <$" + str(p2max)))

            if has_data:
                if is_shifted:
                    axsp[index_row, index_column].errorbar(datapoints[:, 0]
                        , ntu(datapoints[:, 1] - get_shift(datapoints), datapoints[:, 0])[0]
                        , yerr = ntu(datapoints[:, 1] - get_shift(datapoints), datapoints[:, 0])[1] * datapoints[:, 2]
                        , fmt = 'o', capsize = 3, markersize=6, label = 'experiment')
                else:
                    axsp[index_row, index_column].errorbar(datapoints[:, 0], ntu(datapoints[:, 1], datapoints[:, 0])[0]
                        , yerr = ntu(datapoints[:, 1], datapoints[:, 0])[1] * datapoints[:, 2], fmt = 'o', capsize = 3)
            index_column += 1
        index_column = 0
        index_row += 1

    axsp[0, 0].legend(loc=2)
    plt.savefig(figname, bbox_inches='tight')
    print(figname.replace(fig_path, '') + ' DONE!')
    return

def drawDM():  # Draw different cross sections and compare them.
    dPHI = np.linspace(0, 2.*pi, 2 * len(PHI))
    xub = 2*pi-xlb  # x upper bound, x lower bound has been defined in the argument.
    ylb = -0.1  # y lower bound, y upper bound has been defined in the arguments.

    ind_ignore = int(xlb / (2*pi) * len(dPHI))  # Index in list to ignore when ranged is True.
    # Get the name of the figure.
    figname = DMfig_path + 'Diff' + curve_type + '.png'
    for _suffix in DMsuffix:
        figname = figname.replace('.png', '+' + _suffix + '.png')
    if is_relative:
        figname = figname.replace('.png', '_relative.png')
        if A != 1:
            figname = figname.replace('.png', ('_A' + str(int(A)) + '.png'))
    if is_norm:
        figname = figname.replace('.png', '_norm.png')
    if is_ranged:
        dPHI = dPHI[ind_ignore : 2*len(PHI) - ind_ignore]
        figname = figname.replace('.png', ('_ranged' + str(round(xlb, 3)) + '.png'))
    if is_shifted:
        figname = figname.replace('.png', '_shifted.png')
    if has_data:
        figname = figname.replace('.png', '_withdata.png')
    if has_band:
        figname = figname.replace('.png', '_withband.png')
    
    # Draw the figure
    fig = plt.figure(figsize = (16, 16))
    gs = fig.add_gridspec(len(P2s)-1, len(P1s)-1, hspace=0, wspace=0)
    axsp = gs.subplots(sharex=True, sharey=True)
    colors = ['red', 'green', 'blue', 'cyan', 'orange', 'magenta', 'teal', 'yellow']  # Colors for the curves.

    index_row = 0  # Row index of the graph.
    index_column = 0  # Column index of the graph.

    for p1_ind in range(len(P1s)-1):
        for p2_ind in range(len(P2s)-1):
            plt.xlim(xlb, xub)
            plt.ylim(ylb, yub)

            ##### For each bin #####
            p1min = P1s[p1_ind]
            p1max = P1s[p1_ind+1]
            p2min = P2s[p2_ind]
            p2max = P2s[p2_ind+1]
            # Label the curve for its momenta bin.
            if is_norm == True:
                axsp[index_row, index_column].text((xlb+(xub-xlb)*0.75), yub*0.9, (str(p1min) + "$< k_{γt} <$" + str(p1max)))
                axsp[index_row, index_column].text((xlb+(xub-xlb)*0.75), yub*0.8, (str(p2min) + "$< P_{ht} <$" + str(p2max)))

            if has_data:  # Get the data points and draw them if needed.
                datapoints = np.array(read_table(data_path\
                    + str(p1min) + '_' + str(p1max) + '_'\
                    + str(p2min) + '_' + str(p2max) + '.txt', sep = '&'))
                if is_shifted:
                    axsp[index_row, index_column].errorbar(datapoints[:, 0]
                        , ntu(datapoints[:, 1] - get_shift(datapoints), datapoints[:, 0])[0]
                        , yerr = ntu(datapoints[:, 1] - get_shift(datapoints), datapoints[:, 0])[1] * datapoints[:, 2]
                        , fmt = 'o', capsize = 3, markersize=6, label = 'exp data')
                else:
                    axsp[index_row, index_column].errorbar(datapoints[:, 0], ntu(datapoints[:, 1], datapoints[:, 0])[0]
                        , yerr = ntu(datapoints[:, 1], datapoints[:, 0])[1] * datapoints[:, 2], fmt = 'o', capsize = 3)

            _CGClist, _Sudlist, _CGCSudlist = [], [], []
            for _suffix in DMsuffix:  # Get the raw cross sections.
                _CS = np.array(read_table(CS_path + _suffix + '_'\
                    + str(p1min) + '_' + str(p1max) + '_'\
                    + str(p2min) + '_' + str(p2max) + '.txt', sep = '&'))
                if is_ranged:
                    _CGClist.append(_CS[:, 0][ind_ignore : 2*len(PHI) - ind_ignore])
                    _Sudlist.append(_CS[:, 1][ind_ignore : 2*len(PHI) - ind_ignore])
                    _CGCSudlist.append(_CS[:, 2][ind_ignore : 2*len(PHI) - ind_ignore])
                else:
                    _CGClist.append(_CS[:, 0])
                    _Sudlist.append(_CS[:, 1])
                    _CGCSudlist.append(_CS[:, 2])
            _CGClist = np.array(_CGClist)
            _Sudlist = np.array(_Sudlist)
            _CGCSudlist = np.array(_CGCSudlist)
            _CGCpp = _CGClist[0]
            _Sudpp = _CGClist[0]
            _CGCSudpp = _CGClist[0]

            ########## Make modifications to the cross sections if needed, then draw them out ##########
            for CSind in range(len(DMsuffix)): 
                ##### For each cross section #####
                _CGC = _CGClist[CSind]
                _Sud = _Sudlist[CSind]
                _CGCSud = _CGCSudlist[CSind]
                _color = colors[CSind]
                _suffix = DMsuffix[CSind]
                if A != 1:
                    if CSind != 0:
                        _CGC = _CGC/A
                        _Sud = _Sud/A
                        _CGCSud = _CGCSud/A
                if (is_norm and not is_relative) or CSind == 0:  # If normalize==True or it's the pp curve.
                    _CGC = ntu(_CGC, dPHI)[0]
                    _Sud = ntu(_Sud, dPHI)[0]
                    _CGCSud = ntu(_CGCSud, dPHI)[0]
                if is_relative and CSind != 0:
                    _CGC = np.divide(_CGC, _CGCpp, out=np.zeros_like(_CGC), where=_CGCpp!=0) * ntu(_CGCpp, dPHI)[0]
                    _Sud = np.divide(_Sud, _Sudpp, out=np.zeros_like(_Sud), where=_Sudpp!=0) * ntu(_Sudpp, dPHI)[0]
                    _CGCSud = np.divide(_CGCSud, _CGCSudpp, out=np.zeros_like(_CGCSud), where=_CGCSudpp!=0) * ntu(_CGCSudpp, dPHI)[0]
                if is_ratio and CSind != 0:
                    _CGCratio = np.divide(_CGC, _CGCpp, out=np.zeros_like(_CGC), where=_CGCpp!=0)
                    _Sudratio = np.divide(_Sud, _Sudpp, out=np.zeros_like(_Sud), where=_Sudpp!=0)
                    _CGCSudratio = np.divide(_CGCSud, _CGCSudpp, out=np.zeros_like(_CGCSud), where=_CGCSudpp!=0)

                bandmin = []
                bandmax = []
                
                
                ##### Draw the curves #####
                _Clabel = ('CGC_' + _suffix)
                _Slabel = ('Sud_' + _suffix)
                _CSlabel = ('CGC+Sud_' + _suffix)
                for _replace in text_replace:
                    _replace = '_' + _replace
                    _Clabel = _Clabel.replace(_replace, '')
                    _Slabel = _Slabel.replace(_replace, '')
                    _CSlabel = _CSlabel.replace(_replace, '')

                if curve_type == 'C':
                    axsp[index_row, index_column].plot(dPHI, _CGC, color = _color, linewidth = 1, label = _Clabel, alpha = 0.5)
                if curve_type == 'S':
                    axsp[index_row, index_column].plot(dPHI, _Sud, color = _color, linewidth = 1, label = _Slabel, alpha = 0.5)
                if curve_type == 'CS':
                    axsp[index_row, index_column].plot(dPHI, _CGCSud, color = _color, linewidth = 1, label = _CSlabel, alpha = 0.5)
                if curve_type == 'A':
                    axsp[index_row, index_column].plot(dPHI, _CGC, ls = '--', color = _color, linewidth = 1, label = _Clabel, alpha = 0.5)
                    axsp[index_row, index_column].plot(dPHI, _Sud, ls = '-.', color = _color, linewidth = 1, label = _Slabel, alpha = 0.5)
                    axsp[index_row, index_column].plot(dPHI, _CGCSud, color = _color, linewidth = 1, label = _CSlabel, alpha = 0.5)                
            index_row += 1
        index_row = 0
        index_column += 1

    axsp[0, 0].legend(loc=2, fontsize = 10, framealpha = 0.5)
    plt.savefig(figname, bbox_inches='tight')
    print(figname.replace(DMfig_path, '') + ' DONE!')
    return

############################## __main__ ##############################
if __name__ == '__main__':
    if args.option == 'CS':
        getCS()
    elif args.option == 'ALL':
        drawALL()
    elif args.option == 'DM':
        drawDM()