#!/usr/bin/env python3

from numpy import *
from sys import argv, exit
from os import path, mkdir
import h5py
from scipy import interpolate
import shutil
import numpy as np
import pylab as pl
from os import path, mkdir
import os
HBARC = 0.197327053



Phi30 = np.array([-3.14159, -2.94524, -2.74889, -2.55254, -2.35619, -2.15984, -1.9635, -1.76715, -1.5708, -1.37445, -1.1781, -0.981748, -0.785398, -0.589049, -0.392699, -0.19635, 1.11022e-15, 0.19635, 0.392699, 0.589049, 0.785398, 0.981748, 1.1781, 1.37445, 1.5708, 1.76715, 1.9635, 2.15984, 2.35619, 2.55254, 2.74889, 2.94524, 3.14159])
Phi = np.array([-3.14159, -2.94524, -2.74889, -2.55254, -2.35619, -2.15984, -1.9635, -1.76715, -1.5708])

plotname = ["J/$\Psi$, $Q^2$=0.0GeV$^2$","$\\rho$,$Q^2$=0.0GeV$^2$","$\\rho$,$Q^2$=6.6GeV$^2$","$\\rho$,$Q^2$=11.5GeV$^2$","$\\rho$,$Q^2$=17.4GeV$^2$","$\\rho$,$Q^2$=33GeV$^2$"]
try:
    data_path = path.abspath(argv[1])
    data_name = data_path.split("/")[-1]
    results_folder_name = data_name.split(".h5")[0]
    avg_folder_header = path.join(results_folder_name)
    print("output folder: %s" % avg_folder_header)
    """
    if(path.isdir(avg_folder_header)):
        print("folder %s already exists!" % avg_folder_header)
        var = input("do you want to delete it? [y/N]")
        if 'y' in var:
            shutil.rmtree(avg_folder_header)
        else:
            print("please choose another folder path~")
            exit(0)
    mkdir(avg_folder_header)
    """
except IndexError:
    print("Usage: {} results.h5 results_folder".format(argv[0]))
    exit(1)


# analysis begins ...
lcut = 71
hf = h5py.File(data_path, "r")
event_list = list(hf.keys())

name = ["JPsi_Q2_0"]#, "rho_Q2_0"]#,"rho_Q2_6.6","rho_Q2_11.5","rho_Q2_17.4","rho_Q2_33"]
for ip in range(len(name)):
    realpart = 0
    nevent = 0
    t_arr = []; t_arrFlag = True; B_Ncoll = []
    for iev, event_name in enumerate(event_list):
        #event_id = int(event_name.split("_")[-1])
        event_id = int(0)
        event_group = hf.get(event_name)
        for ifile in range(1):
            filename = "rho_Q2_0.0_0_0"
            temp_data1 = event_group.get(filename)
            temp_data1 = nan_to_num(temp_data1)
            if (sum(temp_data1[:,1]) > 5e5):
                filename = "B_Ncoll.dat"
                BNcoll = event_group.get(filename)
                BNcoll = nan_to_num(BNcoll)
                print("weird event", event_name, BNcoll[0], '{:.4e}'.format(sum(temp_data1[:,2])), '{:.4e}'.format(sum(temp_data1[:,3])))
                continue
            try:
                iiii = temp_data1.shape
            except AttributeError as e:
                continue
            try:
                if temp_data1.shape == (0,): continue
            except AttributeError as e:
                 print("empty event: ", event_name)
                 continue
            filename = "B_Ncoll.dat"
            BNcoll = event_group.get(filename)
            BNcoll = nan_to_num(BNcoll)
            B_Ncoll.append(BNcoll[0])
            realpart = realpart + temp_data1
            nevent = nevent +1
            if t_arrFlag:
                t_arr = temp_data1[:, 0]
                t_arrFlag = False
    print("nevent = ", nevent)
    t_arr = array(t_arr)
    realpart = array(realpart)
    realpart = realpart
    B_Ncoll  = array(B_Ncoll)
    print("Mean Bmin in fm ", mean(B_Ncoll))
    
nn = 0
pp = os.listdir("RESULTS_text")
alll = 0

#for iev in range(len(pp)):
#    try:
#        #files = np.loadtxt("RESULTS/{}".format(int(pp[iev])))
#        files = np.genfromtxt("RESULTS_text/{}".format(int(pp[iev])), skip_footer=1)
#    except ValueError as e:
#        print("wrong event", pp[iev])
#        continue
#    else:
#        if (len(files) < lcut):
#            continue
#        else:
#            nn = nn +1
#            alll = alll + files[0:lcut, :]
#print(nn)
alll = alll

nevent = nevent + nn
print("dddddddddddddddddd", nevent)
realpart = realpart[0:lcut, : ] + alll
realpart = realpart/nevent
C0mean = realpart[:,5]
C2mean = realpart[:,6] 
tarr = t_arr[0:lcut]
#dsigma_dtmean = dsigma_dtall/weightall

    
dsigma_dtmean = []
dsigma_dtmean.append(tarr)
for ii in range(len(Phi)):
    mid = 0.5*C0mean + 0.5 * C2mean * np.cos(2.*Phi[ii])
    dsigma_dtmean.append(mid)
dsigma_dtmean = np.array(dsigma_dtmean)
dsigma_dtmean = dsigma_dtmean.T
np.savetxt("dsigma_dtmean", dsigma_dtmean,  fmt="%.6e", delimiter="  ")
ttt = dsigma_dtmean[:,0]
llll = 8
uuuu = 27
mid = ttt[llll:uuuu]

TT = np.arange(0.0034, 0.0127, 0.0002)
dddd = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/IP-Glasma-Diff/UPC_diffractive/AuAu/Figure3C")
dataerr = dddd[:,2]/dddd[:,1]
print(dddd[:,0])
f_coh = interpolate.interp1d(dddd[:,0], log(dataerr + 1e-30), kind="cubic")
dataerr = exp(f_coh(TT))
print(dataerr)

for ij in range(1, len(Phi)+1):
    zz = dsigma_dtmean[:,ij]
    f_coh = interpolate.interp1d(ttt, log(zz + 1e-30), kind="cubic")
    #TT = dddd[:,0]
    coh_data = exp(f_coh(TT))
    zmid = coh_data * 150
    alll = array([TT, zmid, zmid*dataerr/5.])
    np.savetxt("data_itheta_{}".format(ij-1), alll.T)
Phi_diff = []
for ie in range(1):
    C0 = C0mean
    C2 = C2mean
    mean_phiall = dsigma_dtmean[:,1:len(Phi)+1]
    print(mean_phiall.shape)
    
    exec("pl.figure({})".format(0+23))
    pl.yscale('log')
    zz = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/IP-Glasma-Diff/UPC_diffractive/AuAu/Figure3D")
    pl.plot(zz[:,0], zz[:,1],'bh',label='STAR data')
    pl.errorbar(zz[:,0], zz[:,1], zz[:,2],0, capsize=0, ls='none', color='b', elinewidth=2)
    
    zz = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/IP-Glasma-Diff/UPC_diffractive/AuAu/Figure3C")
    pl.plot(zz[:,0], zz[:,1],'rh',label='STAR data')
    pl.errorbar(zz[:,0], zz[:,1], zz[:,2],0, capsize=0, ls='none', color='r', elinewidth=2)
    
    cc = zz[:,1]
    ttt = zz[:,0]
    cc = cc[:][ ttt < 0.01
    ]
    data_sum = np.sum(cc) * (cc[1] - cc[0])
    dd = mean_phiall[1:len(tarr),0]
    tarr1 = tarr[1:len(tarr)]
    dd = dd[:][ tarr1 < 0.01
    ]
    scale =  (np.sum(cc) * (zz[1,0]-zz[0,0])) /(np.sum(dd)* (tarr1[1]-tarr1[0]))
    
    f_coh = interpolate.interp1d(zz[:,0], log(zz[:,1] + 1e-30), kind="cubic")
    #TT = dddd[:,0]
    coh_data = exp(f_coh(TT))
    zmid = coh_data * 150/scale
    
    dataerr = zz[:,2]
    f_coh = interpolate.interp1d(zz[:,0], log(dataerr + 1e-30), kind="cubic")
    dataerr = exp(f_coh(TT))

    alll = array([TT, zmid, dataerr * 150 /scale])
    np.savetxt("data_phi_0", alll.T)
    
    pl.plot(tarr1, scale * mean_phiall[1:len(tarr),-1],'b',label='$\phi_p - \Phi_P$ = $\pi/2$')
    pl.plot(tarr1, scale * mean_phiall[1:len(tarr),0],'r',label='$\phi_p - \Phi_P$ = 0.0')
    xx = scale * mean_phiall[1:len(tarr),-1]
    yy = scale * mean_phiall[1:len(tarr),0]
    aaa = np.array([tarr1, xx, yy])
    print(xx.size, tarr1.size)
    np.savetxt("v2pt", aaa.T)
    #pl.errorbar(mt,cmdn, cmdnerr,0, capsize=0, ls='none', color='g', elinewidth=2)
    print("scale", scale)
    np.savetxt("mean_phiall", mean_phiall[5:len(tarr1), :],  fmt="%.6e", delimiter="  ")
    np.savetxt("tarr1",tarr1)
    pl.legend( loc='upper right' , fontsize=12)
    pl.xlim(0,0.02)
    pl.ylim(1e-4,9e-2)
    pl.title('total, UPC at Au+Au, minus 0.0 fm, $R_{WS}$ = 6.38 fm ', fontsize=15)# give plot a title
    pl.xlabel('|t| [$GeV^2$]', fontsize=12)# make axis labels
    pl.ylabel('$dN/dt$ (arb. norm)', fontsize=12)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=13)
    #pl.show()# show the plot on the screen
    pl.savefig('UPC_diff_rho_data.png', dpi=120)
    
    
    exec("pl.figure({})".format(0+10))
    mean_phi = mean_phiall[1:len(tarr),:]
    zz = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/IP-Glasma-Diff/UPC_diffractive/AuAu/Figure4A_AuAu")
    pl.plot(zz[:,0], zz[:,1],'kh',label='STAR data')
    pl.errorbar(zz[:,0], zz[:,1], zz[:,2],0, capsize=0, ls='none', color='k', elinewidth=2)
    for iphi in range(len(Phi)):
         temp = mean_phi.T[iphi]
         t2 = temp[:][ tarr1 < 0.01
         ]
         Phi_diff.append(np.sum(t2))
    Phi_diff = np.array(Phi_diff)
    Phi_diffmid = []
    for iphi in range(len(Phi)):
        Phi_diffmid.append(Phi_diff[iphi])
    for iphi in range(len(Phi)-2, -1, -1):
        Phi_diffmid.append(Phi_diff[iphi])
    for iphi in range(1, len(Phi)):
        Phi_diffmid.append(Phi_diff[iphi])
    for iphi in range(len(Phi)-2, -1, -1):
        Phi_diffmid.append(Phi_diff[iphi])
    Phi_diff = np.array(Phi_diffmid)
    print(Phi_diff.size)
    scale = 2*math.pi / np.sum(Phi_diff)/(Phi[1]-Phi[0])*1.03
    Phi_diff = Phi_diff * scale
    pl.plot(Phi30, Phi_diff,'b',label='$\\rho$')
    aaa = np.array([Phi30, Phi_diff])
    np.savetxt("integrated_v2_0n0n", aaa.T)
    #pl.errorbar(mt,cmdn, cmdnerr,0, capsize=0, ls='none', color='g', elinewidth=2)
    pl.legend( loc='upper right' , fontsize=12)
    #pl.xlim(0,0.25)
    #pl.ylim(10,1e7)
    pl.title('total, UPC at Pb+Pb at 5.02 TeV, |$q^2_{\perp}$| < 0.01 $GeV^2$', fontsize=12)# give plot a title
    pl.xlabel('$\phi_p - \Phi_P$', fontsize=12)# make axis labels
    pl.ylabel('Counts (norm. to unity)', fontsize=12)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=13)
    #pl.show()# show the plot on the screen
    pl.savefig('UPC_diff_Delta_Phi_rho_scaled.png', dpi=120)
    
    xx = [0,100]
    yy = [0, 0]
    
    exec("pl.figure({})".format(1234655))
    pl.plot(xx, yy,'k--')
    zz = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/IP-Glasma-Diff/UPC_diffractive/AuAu/Figure4B")
    pl.plot(zz[:,0], zz[:,1],'kh',label='STAR data')
    pl.errorbar(zz[:,0], zz[:,1], zz[:,2],0, capsize=0, ls='none', color='k', elinewidth=2)
    cons2phi_v2 = C2/C0
    pl.plot(tarr**0.5, cons2phi_v2,'r',label='$\\rho$')
    bb = np.array([tarr**0.5, cons2phi_v2])
    #np.savetxt("v2pt", bb.T)
    # estimate the integrated v2 
    cons2phi_v2 = cons2phi_v2[1:len(cons2phi_v2)]
    cross_section = dsigma_dtmean[:,1:len(Phi)+1]
    cross_section = cross_section[1:len(tarr):,]
    cross_section_phi_average = np.mean(cross_section, axis = 1)
    estimate_v2 = np.sum(cross_section_phi_average * cons2phi_v2)/np.sum(cross_section_phi_average)
    print("estimate_v2 = ", estimate_v2)
    pl.legend( loc='upper right' , fontsize=12)
    pl.xlim(0,0.22)
    #pl.ylim(-0.1, 0.55)
    #pl.ylim(10,1e7)
    pl.title('C2/C0 (total), minus 0.0 fm, $R_{WS}$ = 6.38 fm ', fontsize=15)# give plot a title
    pl.xlabel('$p_{t}$ [$GeV$]', fontsize=12)# make axis labels
    pl.ylabel('2<cos(2$\phi$)>', fontsize=12)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=13)
    #pl.show()# show the plot on the screen
    pl.savefig('UPC_meancos2p_C0C2.png', dpi=120)
    
    exec("pl.figure({})".format(12344655))
    pl.plot(xx, yy,'k--')
    zz = np.loadtxt("/home/wenbin/Downloads/Wenbin_working/Work/WSU_BNL_work/IP-Glasma-Diff/UPC_diffractive/AuAu/Figure4B")
    pl.plot(zz[:,0]**2, zz[:,1],'kh',label='STAR data')
    pl.errorbar(zz[:,0]**2, zz[:,1], zz[:,2],0, capsize=0, ls='none', color='k', elinewidth=2)
    cons2phi_v2 = C2/C0
    pl.plot(tarr, cons2phi_v2,'r',label='$\\rho$')
    # estimate the integrated v2 
    cons2phi_v2 = cons2phi_v2[1:len(cons2phi_v2)]
    cross_section = dsigma_dtmean[:,1:len(Phi)+1]
    cross_section = cross_section[1:len(tarr):,]
    cross_section_phi_average = np.mean(cross_section, axis = 1)
    estimate_v2 = np.sum(cross_section_phi_average * cons2phi_v2)/np.sum(cross_section_phi_average)
    print("estimate_v2 = ", estimate_v2)
    pl.legend( loc='upper right' , fontsize=12)
    pl.xlim(0,0.03)
    pl.ylim(-0.1, 0.55)
    #pl.ylim(10,1e7)
    pl.title('C2/C0 (total),  ', fontsize=15)# give plot a title
    pl.xlabel('$|t|$ [$GeV^{2}$]', fontsize=12)# make axis labels
    pl.ylabel('2<cos(2$\phi$)>', fontsize=12)
    pl.xticks(fontsize=12)
    pl.yticks(fontsize=13)
    #pl.show()# show the plot on the screen
    pl.savefig('UPC_meancos2p_C0C2t.png', dpi=120)
    
    
hf.close()
"""
# interpolate the coherent/incoherent cross section to the exp. data points
f_coh = interpolate.interp1d(t_arr, log(coherent + 1e-30), kind="cubic")
f_coh_err = interpolate.interp1d(t_arr, log(coherent_err + 1e-30),
                                 kind="cubic")
f_incoh = interpolate.interp1d(t_arr, log(incoherent + 1e-30), kind="cubic")
f_incoh_err = interpolate.interp1d(t_arr, log(incoherent_err + 1e-30),
                                   kind="cubic")
TT = t_data_coherent
coh_data = exp(f_coh(TT))
coh_data_err = exp(f_coh_err(TT))

TT = t_data_incoherent[0:7]  # only consider 0-2.5 GeV 
incoh_data = exp(f_incoh(TT))
incoh_data_err = exp(f_incoh_err(TT))

t_arr = concatenate((t_data_incoherent[0:7], t_data_coherent))
model_result = concatenate((incoh_data, coh_data))
model_err = concatenate((incoh_data_err, coh_data_err))
savetxt(path.join(avg_folder_header, "Bayesian_output.txt"),
        array([t_arr, model_result, model_err]).transpose(),
        fmt="%.6e", delimiter="  ",
        header="t  results  stat. err")
"""
