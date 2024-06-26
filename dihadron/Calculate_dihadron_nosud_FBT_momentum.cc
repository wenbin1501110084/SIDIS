#include <stdio.h>
#include <cuba.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <cstdio>
#include <cmath>
#include <math.h>
#include <iomanip>
#include <chrono>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_deriv.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>
#include <complex>
#include <vector>
#include <LHAPDF/LHAPDF.h>
#include "FBT.h"

std::vector<double> YValues, pTValues, F1qgV, F2qgV, F1ggV, F3ggV, F6ggV, FadjV;
const int Klength = 700;
const double minpT = 1.e-05;
const double maxpT = 99.;
const double Y_step = 0.2;
const double ST = 1.0;//M_PI * 6.6*6.6 /0.197/0.197;// GeV^-2
const double CA = 3.;
const double CF = 4./3.;
const double betaSud = 0.7192823;
const double Lambda2Sud = 0.03745153;
double Lambda = sqrt(Lambda2Sud);
const double bmax2 = 2.25;//GeV^-2
const double Q0 = 1.5491933384829668;//GeV
const double g1 = 0.212;
const double g2 = 0.84;
const double rootsnn = 200.;// GeV
const double Y1max = 4.0;
const double Y1min = 2.4;
const double Y2max = 4.0;
const double Y2min = 2.4;
const double Subcoff = 1.0;    
const double Rcut = 0.4;
const int num_threads = 10;
const long long int sample_points = 1000000;

LHAPDF::PDF* FF;
LHAPDF::PDF* pdf;

struct PARAMETERS {
    double etagmax;// GeV
    double etagmin;// GeV
    double Q2;// GeV^2
    double b;// 
    double kpmagmax;// GeV
    double etahmax;// GeV
    double etahmin;// GeV
    double kTgmax;// GeV
    double kTgmin;// GeV
    double PhTmax;// GeV
    double PhTmin;// GeV
    double qqTT;// GeV
    double bmax;
    int iiyy;
    double y1max;
    double y1min;
    double y2max;
    double y2min;
    double p1magmax;
    double p1magmin;
    double p2magmin;
    double p2magmax;
    double Delta_phi;
};

using namespace std;

double alphas(double Q2) {
    double Lambda2 = 0.197*0.197; // GeV^2
    double alphas = 4.*M_PI /((11. - 8./3.)*log(Q2/Lambda2));
    return alphas;
}


double Sud_qgqg(double Q, double bmag) {
    double bstar = bmag / sqrt(1. + bmag*bmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    if (mub > Q) return 0.0;
    double A = 1.37929353;
    double B = -1.35269822;
    double Sudp = 1/betaSud * (-2.*A*log(Q/mub) + (B + 2.*A*log(Q/Lambda)) * (log((log(Q/Lambda)) / (log(mub/Lambda)))));
    double Sudnp = (2. + CA/CF) * g1 / 2. * bmag*bmag + (2.+CA/CF)*g2/2. * log(Q/Q0)*log(bmag/bstar);
    double Sud = Sudp + Subcoff * Sudnp;
    return Sud;
}

double Sud_gggg(double Q, double bmag){
    double bstar = bmag / sqrt(1. + bmag*bmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    if (mub > Q) return 0.0;
    double A = 1.909781;  
    double B = -2.14840305;
    double Sudp = 1/betaSud * (-2.*A*log(Q/mub) + (B + 2.*A*log(Q/Lambda)) * (log((log(Q/Lambda)) / (log(mub/Lambda)))));
    double Sudnp = 3. * CA/CF *g1/2.*bmag*bmag + 3.*CA/CF*g2/2.*log(Q/Q0) *log(bmag/bstar);
    double Sud = Sudp + Subcoff * Sudnp;
    return Sud;
}

double Wq2qg(double bpmag, double z1, double z2, double xq, double Q2) {
    double bstar = bpmag / sqrt(1. + bpmag*bpmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    double mub2 = mub * mub;
    if (std::isnan(mub2)) {
        mub2 = Q2;
        std::cout << "Warning: myValue is NaN " << bstar << "  " << bpmag << std::endl;
    } 
    if (mub2 > Q2) mub2= Q2;
    if (mub2 < 1.0) mub2 = 1.0;
    double Q = sqrt(Q2);
    double Dqxqx1 = 3.*FF->xfxQ2(2, z1, mub2)/z1 * pdf->xfxQ2(2,xq, mub2) + 3.*FF->xfxQ2(1, z1, mub2)/z1 * pdf->xfxQ2(1,xq, mub2);
    double bWbt = 2.*M_PI * bpmag * Dqxqx1 * FF->xfxQ2(21, z2, mub2)/z2 * exp(-1.*Sud_qgqg(Q, bpmag));
    return bWbt;
}

double Wg2qq(double bpmag, double z1, double z2, double xq, double Q2) {
    double bstar = bpmag / sqrt(1. + bpmag*bpmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    double mub2 = mub * mub;
    if (std::isnan(mub2)) {
        mub2 = Q2;
        std::cout << "Warning: myValue is NaN " << bstar << "  " << bpmag << std::endl;
    } 
    if (mub2 > Q2) mub2= Q2;
    if (mub2 < 1.0) mub2 = 1.0;
    double Q = sqrt(Q2);
    double Dqxqx1 = FF->xfxQ2(2, z1, mub2)/z1 * FF->xfxQ2(2, z2, mub2)/z2 + FF->xfxQ2(1, z1, mub2)/z1 * FF->xfxQ2(1, z2, mub2)/z2 +
                    FF->xfxQ2(3, z1, mub2)/z1 * FF->xfxQ2(3, z2, mub2)/z2;
    double bWbt = 2.*M_PI * bpmag * Dqxqx1 * pdf->xfxQ2(21,xq, mub2) * exp(-1.*Sud_qgqg(Q, bpmag));
    return bWbt *2.;
}

double Wg2gg(double bpmag, double z1, double z2, double xq, double Q2) {
    double bstar = bpmag / sqrt(1. + bpmag*bpmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    double mub2 = mub * mub;
    if (std::isnan(mub2)) {
        mub2 = Q2;
        std::cout << "Warning: myValue is NaN " << bstar << "  " << bpmag << std::endl;
    } 
    if (mub2 > Q2) mub2= Q2;
    double Q = sqrt(Q2);
    if (mub2 < 1.0) mub2 = 1.0;
    double Dqxqx1 = FF->xfxQ2(21, z1, mub2)/z1;
    double bWbt = 2.*M_PI * bpmag * Dqxqx1 * FF->xfxQ2(21, z2, mub2)/z2 * pdf->xfxQ2(21,xq, mub2) * exp(-1.*Sud_gggg(Q, bpmag));
    return bWbt*2.;
}


int zh_kp2_b_integrated(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    double p1mag = x[2] * (helper->p1magmax - helper->p1magmin) + helper->p1magmin;
    double thetap1 = x[3] * 2. * M_PI;
    double pt2maxtep = std::min(helper->p2magmax, p1mag);
    double p2mag = x[4] * (pt2maxtep - helper->p2magmin) + helper->p2magmin;
    double y1 = x[5] * (helper->y1max - helper->y1min) + helper->y1min;
    double y2 = x[6] * (helper->y2max - helper->y2min) + helper->y2min;
    
    
    double xgd1 = p1mag *exp(y1)/rootsnn / (1. - p2mag*exp(y2)/rootsnn);
    
    double z1 = x[0] * (1.-xgd1) + xgd1;
    
    double xgd2 = p2mag *exp(y2)/rootsnn / (1. - p1mag/z1*exp(y1)/rootsnn);
    double z2 = x[1] * (1.-xgd2) + xgd2;
    
     //double z1 = x[0];
     //double z2 = x[1];
     
    if (xgd1 > 1.0  || xgd2>1.0 ) {
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
        f[3] = 0.0;
        return 0;
    }
    
    //double kpmag = x[7] * helper->kpmagmax;
    //double thetakp = x[8] * 2. * M_PI;
    /*
    double R2condition = (y1 - y2) * (y1 - y2) + helper->Delta_phi * helper->Delta_phi;
    if (R2condition < Rcut*Rcut) {
        f[0] = 0.0;
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
        f[3] = 0.0;
        return 0;
    }
    */
    double thetap2 = thetap1 - helper->Delta_phi;
    double total_volume = (helper->y1max - helper->y1min) * (helper->y2max - helper->y2min) * (helper->p1magmax - helper->p1magmin) *
                          (pt2maxtep - helper->p2magmin) *  2. * M_PI  * (1.-xgd1) * (1.-xgd2);
    double kT1 = p1mag/z1;
    double kTx1 = kT1 * cos(thetap1); 
    double kTy1 = kT1 * sin(thetap1); 
    
    double kT2 = p2mag/z2;
    double kTx2 = kT2 * cos(thetap2); 
    double kTy2 = kT2 * sin(thetap2); 
    double qTx = kTx1 + kTx2;
    double qTy = kTy1 + kTy2;
    double qT = sqrt(qTx*qTx + qTy*qTy);
    
    //double kTxdiff = kpmag*cos(thetakp) - qTx;
    //double kTydiff = kpmag*sin(thetakp) - qTy;
    //double kminuskp_mag = sqrt(pow(kTxdiff, 2) + pow(kTydiff, 2));
    double xq = std::max(kT1, kT2) * (exp(y1) + exp(y2))/rootsnn;
    double xg = std::max(kT1, kT2) * (exp(-y1)  + exp(-y2))/rootsnn;
    if (xq > 1.0 || xg > 1.) {
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
        f[3] = 0.0;
        return 0;
    }
    //double Q2 = xq * xg * rootsnn * rootsnn;
    double zm1 = kT1 * exp(y1) / (kT1 * exp(y1) + kT2 * exp(y2));
    double PTtemp = pow((1.-zm1) * kTx1 - zm1 * kTx2, 2.) + pow((1.-zm1) * kTy1 - zm1 * kTy2, 2.);
    double PT1 = sqrt(PTtemp);
    double rapidity = log(0.01/xg);
    if (rapidity< 0.0) rapidity = 0.0;
    int y_index = int(rapidity/Y_step);
    if (qT > maxpT) qT = maxpT;
    if (qT < minpT) qT = minpT;
    //FBT ogata0 = FBT(0.0, 0, 1000); // Fourier Transform with Jnu, nu=0.0 and N=10
    
    //double Wq2qgK1 = ogata0.fbt(std::bind(Wq2qg, std::placeholders::_1, z1, z2, xq, Q2), kminuskp_mag);
    //double Wq2qgK2 = ogata0.fbt(std::bind(Wq2qg, std::placeholders::_1, z2, z1, xq, Q2), kminuskp_mag);
    //double Wg2qqK = ogata0.fbt(std::bind(Wg2qq, std::placeholders::_1, z1, z2, xq, Q2), kminuskp_mag);
    //double Wg2ggK = ogata0.fbt(std::bind(Wg2gg, std::placeholders::_1, z1, z2, xq, Q2), kminuskp_mag);
    
    std::vector<double> xValues, F1qgVt, F2qgVt, F1ggVt, F3ggVt, FadjVt;
    xValues.clear(); F1qgVt.clear(); F2qgVt.clear(); F1ggVt.clear(); F3ggVt.clear(); FadjVt.clear();  
    for (int inn= y_index * Klength; inn < y_index * Klength + Klength; inn++) {
        xValues.push_back(pTValues[inn]);
        F1qgVt.push_back(F1qgV[inn]);
        F2qgVt.push_back(F2qgV[inn]);
        F1ggVt.push_back(F1ggV[inn]);
        F3ggVt.push_back(F3ggV[inn]);
        FadjVt.push_back(FadjV[inn]);
    }
    
    gsl_interp *interp1 = gsl_interp_alloc(gsl_interp_cspline, Klength);
    
    gsl_interp_init(interp1, xValues.data(), F1qgVt.data(), Klength);
    double inter_F1qgVt = gsl_interp_eval(interp1, xValues.data(), F1qgVt.data(), qT, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), F2qgVt.data(), Klength);
    double inter_F2qgVt = gsl_interp_eval(interp1, xValues.data(), F2qgVt.data(), qT, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), F1ggVt.data(), Klength);
    double inter_F1ggVt = gsl_interp_eval(interp1, xValues.data(), F1ggVt.data(), qT, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), F3ggVt.data(), Klength);
    double inter_F3ggVt = gsl_interp_eval(interp1, xValues.data(), F3ggVt.data(), qT, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), FadjVt.data(), Klength);
    double inter_FadjVt = gsl_interp_eval(interp1, xValues.data(), FadjVt.data(), qT, nullptr);
    
    gsl_interp_free(interp1);
    
    xValues.clear(); F1qgVt.clear(); F2qgVt.clear(); F1ggVt.clear(); F3ggVt.clear(); FadjVt.clear();  
    y_index = y_index +1;
    for (int inn= y_index * Klength; inn < y_index * Klength + Klength; inn++) {
        xValues.push_back(pTValues[inn]);
        F1qgVt.push_back(F1qgV[inn]);
        F2qgVt.push_back(F2qgV[inn]);
        F1ggVt.push_back(F1ggV[inn]);
        F3ggVt.push_back(F3ggV[inn]);
        FadjVt.push_back(FadjV[inn]);
    }
    
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_cspline, Klength);
    
    gsl_interp_init(interp2, xValues.data(), F1qgVt.data(), Klength);
    double inter_F1qgVt2 = gsl_interp_eval(interp2, xValues.data(), F1qgVt.data(), qT, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), F2qgVt.data(), Klength);
    double inter_F2qgVt2 = gsl_interp_eval(interp2, xValues.data(), F2qgVt.data(), qT, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), F1ggVt.data(), Klength);
    double inter_F1ggVt2 = gsl_interp_eval(interp2, xValues.data(), F1ggVt.data(), qT, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), F3ggVt.data(), Klength);
    double inter_F3ggVt2 = gsl_interp_eval(interp2, xValues.data(), F3ggVt.data(), qT, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), FadjVt.data(), Klength);
    double inter_FadjVt2 = gsl_interp_eval(interp2, xValues.data(), FadjVt.data(), qT, nullptr);
    
    gsl_interp_free(interp2);
    
    inter_F1qgVt = (inter_F1qgVt * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F1qgVt2 * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_F2qgVt = (inter_F2qgVt * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F2qgVt2 * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_F1ggVt = (inter_F1ggVt * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F1ggVt2 * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_F3ggVt = (inter_F3ggVt * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F3ggVt2 * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_FadjVt = (inter_FadjVt * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_FadjVt2 * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    if (xg>0.01) {
        double ratio = pow(1.-xg, 4.) / 0.96059601; // 0.96059601 = (1-0.01)^4
        inter_F1qgVt =  ratio * inter_F1qgVt;
        inter_F2qgVt =  ratio * inter_F2qgVt;
        inter_F1ggVt =  ratio * inter_F1ggVt;
        inter_F3ggVt =  ratio * inter_F3ggVt;
        inter_FadjVt =  ratio * inter_FadjVt;
    }
    double prefactor = p1mag * p2mag /z1/z1/z2/z2;
    double mub2 = p1mag * p1mag;
    double alpha_s = pdf->alphasQ2(mub2);
    //double alpha_s = alphas(mub2);
    double Hqg2qg1 = alpha_s /2./pow(PT1, 4.0) * (1. + (1.-zm1)*(1.-zm1)) * (1.-zm1); // one alpha_s is canceled by F
    double Hqg2qg2 = alpha_s /2./pow(PT1, 4.0) * (1. + zm1*zm1) * zm1; // one alpha_s is canceled by F
    double Hgg2qq  = alpha_s /6. / pow(PT1, 4.) * zm1*(1.-zm1) * (zm1*zm1+(1.-zm1)*(1.-zm1)); // one alpha_s is canceled by F
    double Hgg2gg  = 2.*alpha_s / pow(PT1, 4.) *(zm1*zm1 + (1.-zm1) * (1.-zm1) + zm1*zm1 * (1.-zm1)*(1.-zm1)); // one alpha_s is canceled by F
    
    double DFuud1 = FF->xfxQ2(2, z2, mub2)/z2 * pdf->xfxQ2(2,xq, mub2) + FF->xfxQ2(1, z2, mub2)/z2 * pdf->xfxQ2(1,xq, mub2);
    double DFuudDg1 = DFuud1 * FF->xfxQ2(21, z1, mub2)/z1;
    double DFuud2 = FF->xfxQ2(2, z1, mub2)/z1 * pdf->xfxQ2(2,xq, mub2) + FF->xfxQ2(1, z1, mub2)/z1 * pdf->xfxQ2(1,xq, mub2);
    double DFuudDg2 = DFuud2 * FF->xfxQ2(21, z2, mub2)/z2;
    double TwofgDqDq = 2.*pdf->xfxQ2(21,xq, mub2) * (
                       FF->xfxQ2(2, z1, mub2)/z1 * FF->xfxQ2(2, z2, mub2)/z2 + FF->xfxQ2(1, z1, mub2)/z1 * FF->xfxQ2(1, z2, mub2)/z2
                       );
    double DFggg = FF->xfxQ2(21, z1, mub2)/z1 * pdf->xfxQ2(21,xq, mub2) * FF->xfxQ2(1, z2, mub2)/z2;
    
    double dsigmaqgqg_dDeltaphi = prefactor * DFuudDg1 * ( (1.-zm1)*(1.-zm1)*inter_F1qgVt + inter_F2qgVt ) * Hqg2qg1 +
                                  prefactor * DFuudDg2 * (zm1*zm1* inter_F1qgVt + inter_F2qgVt) * Hqg2qg2;
                                  
    double dsigmaggqq_dDeltaphi = prefactor * TwofgDqDq * (inter_F1ggVt - 2.*zm1*(1.-zm1)* inter_FadjVt) * Hgg2qq;
    
    double dsigmagggg_dDeltaphi = prefactor * DFggg * (inter_F1ggVt - 2.*zm1*(1.-zm1) * inter_FadjVt + inter_F3ggVt) * Hgg2gg;
    
    /*
    if (dsigmaqgqg_dDeltaphi < 0.0) {
        dsigmaqgqg_dDeltaphi = 0.0;
    }
    
    if (dsigmaggqq_dDeltaphi < 0.0) {
        dsigmaggqq_dDeltaphi = 0.0;
    }
    
    if (dsigmagggg_dDeltaphi < 0.0) {
        dsigmagggg_dDeltaphi = 0.0;
    }
    */
    
    f[0] = dsigmaqgqg_dDeltaphi * total_volume;
    f[1] = dsigmagggg_dDeltaphi * total_volume;
    f[2] = dsigmaggqq_dDeltaphi * total_volume;
    return 0;
}


int Calculate_trigger(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    double p1mag = x[1] * (helper->p1magmax - helper->p1magmin) + helper->p1magmin;
    //double thetap1 = x[3] * 2. * M_PI;
    double y1 = x[2] * (helper->y1max - helper->y1min) + helper->y1min;
    
    double zd1 = p1mag * exp(y1)/rootsnn;
    double z1 = x[0] * (1.-zd1) + zd1;
    
    if (zd1 > 1.) {
        f[0] = 0.0;
        return 0;
    }
    
    //double z1 = x[0];
    double total_volume = (helper->y1max - helper->y1min)  * (helper->p1magmax - helper->p1magmin) * 2. * M_PI * (1.-zd1);
    double kT1 = p1mag/z1;
    
    double xq = kT1 * exp(y1)/rootsnn;
    double xg = kT1 * exp(-y1)/rootsnn;
    if (xq > 1.0 || xg > 1.) {
        f[0] = 0.0;
        return 0;
    }
    double Q2 = p1mag * p1mag;
    
    double rapidity = log(0.01/xg);
    if (rapidity< 0.0) rapidity = 0.0;
    int y_index = int(rapidity/Y_step);
    if (kT1 > maxpT) kT1 = maxpT;
    if (kT1 < minpT) kT1 = minpT;
    std::vector<double> xValues, F1qgVt, FadjVt;
    xValues.clear(); F1qgVt.clear(); FadjVt.clear();  
    for (int inn= y_index * Klength; inn < y_index * Klength + Klength; inn++) {
        xValues.push_back(pTValues[inn]);
        F1qgVt.push_back(F1qgV[inn]);
        FadjVt.push_back(FadjV[inn]);
    }
    
    gsl_interp *interp1 = gsl_interp_alloc(gsl_interp_cspline, Klength);
    
    gsl_interp_init(interp1, xValues.data(), F1qgVt.data(), Klength);
    double inter_F1qgVt = gsl_interp_eval(interp1, xValues.data(), F1qgVt.data(), kT1, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), FadjVt.data(), Klength);
    double inter_FadjVt = gsl_interp_eval(interp1, xValues.data(), FadjVt.data(), kT1, nullptr);
    
    gsl_interp_free(interp1);
    
    xValues.clear(); F1qgVt.clear(); FadjVt.clear();  
    y_index = y_index +1;
    for (int inn= y_index * Klength; inn < y_index * Klength + Klength; inn++) {
        xValues.push_back(pTValues[inn]);
        F1qgVt.push_back(F1qgV[inn]);
        FadjVt.push_back(FadjV[inn]);
    }
    
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_cspline, Klength);
    
    gsl_interp_init(interp2, xValues.data(), F1qgVt.data(), Klength);
    double inter_F1qgVt2 = gsl_interp_eval(interp2, xValues.data(), F1qgVt.data(), kT1, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), FadjVt.data(), Klength);
    double inter_FadjVt2 = gsl_interp_eval(interp2, xValues.data(), FadjVt.data(), kT1, nullptr);
    
    gsl_interp_free(interp2);
    
    inter_F1qgVt = (inter_F1qgVt * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F1qgVt2 * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_FadjVt = (inter_FadjVt * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_FadjVt2 * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    if (xg>0.01) {
        double ratio = pow(1.-xg, 4.) / 0.96059601; // 0.96059601 = (1-0.01)^4
        inter_F1qgVt =  ratio * inter_F1qgVt;
        inter_FadjVt =  ratio * inter_FadjVt;
    }
    double prefactor = p1mag /z1/z1;
    double alpha_s = pdf->alphasQ2(Q2);
    //double alpha_s = alphas(mub2);
    inter_F1qgVt = 2.*M_PI*M_PI * alpha_s * inter_F1qgVt / 3. / kT1 / kT1;
    inter_FadjVt = 2.*M_PI*M_PI * alpha_s * inter_FadjVt / CF / kT1 / kT1;
    double sumq =  pdf->xfxQ2(2,xq, Q2) * FF->xfxQ2(2, z1, Q2)/z1 + pdf->xfxQ2(1,xq, Q2) * FF->xfxQ2(1, z1, Q2)/z1;
    double sumg =  pdf->xfxQ2(21,xq, Q2) * FF->xfxQ2(21, z1, Q2)/z1;
    double dsigmaqgqg_dDeltaphi = prefactor * (sumq * inter_F1qgVt + sumg*inter_FadjVt);
    if (dsigmaqgqg_dDeltaphi < 0.0) dsigmaqgqg_dDeltaphi = 0.0;
    f[0] = dsigmaqgqg_dDeltaphi * total_volume;
    return 0;
}






int main(int argc, char* argv[]) 
{
    // Set the number of threads to use
    omp_set_num_threads(num_threads);
    
    PARAMETERS params;
    int startid = std::stoi(argv[1]);
    int endid = std::stoi(argv[2]);
    int stage = std::stoi(argv[3]);
    double P1magmin = std::stod(argv[4]);
    double P1magmax = std::stod(argv[5]);
    double P2magmin = std::stod(argv[6]);
    double P2magmax = std::stod(argv[7]); 
                          
    params.y1max    = Y1max;
    params.y1min    = Y1min;
    params.y2max    = Y2max;
    params.y2min    = Y2min;
    params.p1magmax    = P1magmax;
    params.p1magmin    = P1magmin;
    params.p2magmin    = P2magmin;
    params.p2magmax    = P2magmax;
    params.kpmagmax    = 100.0;// GeV/c
    
    // Open the input file
    std::ifstream inputFile("Paul_table/Regularged_FT_large_Nuleus_MV_Paul");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    // Read data from the file
    double Y, pTm, F1qgm, F2qgm, F1ggm, F3ggm, F6ggm, Fadjm;
    while (inputFile >> Y >> pTm >> F1qgm >> F2qgm >> F1ggm >> F3ggm >> F6ggm >> Fadjm) {
        YValues.push_back(Y);
        pTValues.push_back(pTm);
        F1qgV.push_back(F1qgm);
        F2qgV.push_back(F2qgm);
        F1ggV.push_back(F1ggm);
        F3ggV.push_back(F3ggm);
        F6ggV.push_back(F6ggm);
        FadjV.push_back(Fadjm);
    }
    inputFile.close();
    
        /* Define the integration parameters */
    const int ndim2 = 3; /* number of dimensions */
    const int ncomp2 = 1; /* number of components */
    cout << "Starts " <<endl;
    const long long int mineval = sample_points; /* minimum number of integrand evaluations */
    cout <<"step1" <<endl;
    const long long int nvec = 1; /* minimum number of integrand evaluations */
    const cubareal epsrel = 1e-3; /* relative error tolerance */
    const cubareal epsabs = 1e-3; /* absolute error tolerance */
    //const int verbose = 0; /* verbosity level */
    const long long int maxeval = sample_points; /* maximum number of integrand evaluations */
    const long long int nstart = sample_points;
    const long long int nincrease = sample_points;
    const long long int nbatch = sample_points;
    const int gridno = 0;
    const int flags = 0; 
    const int seed = 0; 
    /* Allocate memory for the results */
    long long int neval; /* number of integrand evaluations */
    int fail; /* status flag */
    cubareal Trigger[ncomp2]; /* integral estimates */
    cubareal error2[ncomp2]; /* error estimates */
    cubareal prob2[ncomp2]; /* CHI^2 probabilities */
    std::stringstream filename;
    filename << "dSigma_dDeltaPhi_dihadron_noSub_" << startid << "_" << endid << "_" <<  stage 
             << "_"<< P1magmin << "_" << P1magmax << "_" << P2magmin << "_" << P2magmax <<".txt";
    ofstream realA(filename.str());
    
    LHAPDF::initPDFSet("JAM20-SIDIS_PDF_proton_nlo");
    pdf = LHAPDF::mkPDF("JAM20-SIDIS_PDF_proton_nlo", 0);
    const LHAPDF::PDFSet set("JAM20-SIDIS_PDF_proton_nlo"); // arxiv: 2101.04664
    LHAPDF::initPDFSet("JAM20-SIDIS_FF_pion_nlo");
    FF = LHAPDF::mkPDF("JAM20-SIDIS_FF_pion_nlo", 0);  // Use the appropriate member index if there are multiple sets
    
    llVegas(ndim2, ncomp2, Calculate_trigger, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, Trigger, error2, prob2);
    cout << Trigger[0] << endl;
    /* Define the integration parameters */
    const int ndim = 7; /* number of dimensions */
    const int ncomp = 3; /* number of components */
    /* Allocate memory for the results */
    cubareal integral[ncomp]; /* integral estimates */
    cubareal error[ncomp]; /* error estimates */
    cubareal prob[ncomp]; /* CHI^2 probabilities */
    
    int length;
    double detal_theta; double detal_theta_step;
    if (stage == 0) {
    length = 6;
    realA << "# Delta_phi   Wk  sigmahat  Ntidle  dSigma_dDeltaPhi";
    realA << endl;
    
    detal_theta_step = 0.1;
    for (int itheta=startid ; itheta<endid; itheta++) {
        params.Delta_phi = itheta * 1. * detal_theta_step; 
        realA << params.Delta_phi << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] <<  "  " << integral[2] <<  "  " <<  "  " << Trigger[0];
        realA << endl;
    }
    }
    if(stage ==1) {
    detal_theta =  0.5;//params.Delta_phi;
    length = 15;
    detal_theta_step = 0.05;
    for (int itheta= startid+1; itheta<endid +1; itheta++) {
        
        params.Delta_phi = itheta * 1. * detal_theta_step + detal_theta; 
        realA << params.Delta_phi << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] <<  "  " << integral[2] <<  "  " <<   "  "  << Trigger[0];
        realA << endl;
    }
    }
    if (stage == 2) {
    detal_theta = 1.25; //params.Delta_phi;
    length = 50;
    detal_theta_step = (M_PI - detal_theta) / length;
    for (int itheta=1+ startid; itheta<endid +1; itheta++) {
        params.Delta_phi = itheta * 1. * detal_theta_step + detal_theta; 
        realA << params.Delta_phi << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] <<  "  " << integral[2] <<  "  " <<  "  " << Trigger[0];
                cout << (integral[0] + integral[1] + integral[2] )/Trigger[0] << endl;
        realA << endl;
    }
    }   
    realA.close();
}

