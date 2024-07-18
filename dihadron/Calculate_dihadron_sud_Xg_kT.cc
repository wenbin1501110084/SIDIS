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
const double betaSud = 0.68714457;
const double Lambda2Sud = 0.03006785;
double Lambda = sqrt(Lambda2Sud);
const double bmax2 = 0.25;//GeV^-2
const double Q0 = 1.5491933384829668;//GeV
const double g1 = 0.212;
const double g2 = 0.84;
const double rootsnn = 5020.;// GeV
const double Y1max = 4.4;
const double Y1min = 1.5;
const double Y2max = 4.4;
const double Y2min = 1.5;
const double Rcut = 0.4;
const int num_threads = 10;
const long long int sample_points = 2000000;

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

double Sud_qgqg(double Q, double bmag, double Subcoff) {
    double bstar = bmag / sqrt(1. + bmag*bmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    //if (mub > Q) return 0.0;
    //if (mub < 1.0) mub = 1.0;
    if (mub > Q) mub = Q;
    double A =  1.37929241;
    double B =  -1.29970512; // Fit the 1805.05712 Sud_p
    double Sudp = 1/betaSud * (-2.*A*log(Q/mub) + (B + 2.*A*log(Q/Lambda)) * (log((log(Q/Lambda)) / (log(mub/Lambda)))));
    double Sudnp = (2. + CA/CF) * g1 / 2. * bmag*bmag + (2.+CA/CF)*g2/2. * log(Q/Q0)*log(bmag/bstar);
    //if (mub > Q) Sudp = 0.0;
    //if (mub < 1.0) Sudp = 0.0;
    //if (Sudp < 0.0) Sudp =0.0;
    double Sud = Sudp + Subcoff * Sudnp;
    //if (Sud <0.) Sud = 0.;
    return Sud;
}

double Sud_gggg(double Q, double bmag, double Subcoff){
    double bstar = bmag / sqrt(1. + bmag*bmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    //if (mub > Q) return 0.0;
    //if (mub < 1.0) mub = 1.0;
    if (mub > Q) mub = Q;
    double A = 1.90978212;  
    double B = -1.98934458;  // Fit the 1805.05712 Sud_p
    double Sudp = 1/betaSud * (-2.*A*log(Q/mub) + (B + 2.*A*log(Q/Lambda)) * (log((log(Q/Lambda)) / (log(mub/Lambda)))));
    double Sudnp = 3. * CA/CF *g1/2.*bmag*bmag + 3.*CA/CF*g2/2.*log(Q/Q0) *log(bmag/bstar);
    //if (mub > Q) Sudp = 0.0;
    //if (mub < 1.0) Sudp = 0.0;
    //if (Sudp < 0.0) Sudp =0.0;
    double Sud = Sudp + Subcoff * Sudnp;
    //if (Sud <0.) Sud = 0.;
    return Sud;
}

double Wq2qg(double bpmag, double z1, double z2, double xq, double Q2, double Subcoff) {
    double bstar = bpmag / sqrt(1. + bpmag*bpmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    double mub2 = mub * mub;
    if (std::isnan(mub2)) {
        mub2 = Q2;
        std::cout << "Warning: myValue is NaN " << bstar << "  " << bpmag << std::endl;
    } 
    if (mub2 > Q2) mub2= Q2;
    //if (mub2 < 1.) return 0.0;
    double Q = sqrt(Q2);
    double Dqxqx1 = FF->xfxQ2(2, z2, mub2)/z2 * pdf->xfxQ2(2,xq, mub2) + FF->xfxQ2(1, z2, mub2)/z2 * pdf->xfxQ2(1,xq, mub2);
    double bWbt = 2.*M_PI * bpmag * Dqxqx1 * FF->xfxQ2(21, z1, mub2)/z1 * exp(-1.*Sud_qgqg(Q, bpmag, Subcoff));
    //if (bWbt < 0.) bWbt = 0.;
    return bWbt;
}

double Wg2qq(double bpmag, double z1, double z2, double xq, double Q2, double Subcoff) {
    double bstar = bpmag / sqrt(1. + bpmag*bpmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    double mub2 = mub * mub;
    if (std::isnan(mub2)) {
        mub2 = Q2;
        std::cout << "Warning: myValue is NaN " << bstar << "  " << bpmag << std::endl;
    } 
    if (mub2 > Q2) mub2= Q2;
    //if (mub2 < 1.0) return 0.0;
    double Q = sqrt(Q2);
    double Dqxqx1 = FF->xfxQ2(2, z1, mub2)/z1 * FF->xfxQ2(2, z2, mub2)/z2 + FF->xfxQ2(1, z1, mub2)/z1 * FF->xfxQ2(1, z2, mub2)/z2 +
                    FF->xfxQ2(3, z1, mub2)/z1 * FF->xfxQ2(3, z2, mub2)/z2;
    double bWbt = 2.*M_PI * bpmag * Dqxqx1 * pdf->xfxQ2(21,xq, mub2) * exp(-1.*Sud_qgqg(Q, bpmag, Subcoff));
    //if (bWbt < 0.) bWbt = 0.;
    return bWbt;
}

double Wg2gg(double bpmag, double z1, double z2, double xq, double Q2, double Subcoff) {
    double bstar = bpmag / sqrt(1. + bpmag*bpmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    double mub2 = mub * mub;
    if (std::isnan(mub2)) {
        mub2 = Q2;
        std::cout << "Warning: myValue is NaN " << bstar << "  " << bpmag << std::endl;
    } 
    if (mub2 > Q2) mub2= Q2;
    double Q = sqrt(Q2);
   // if (mub2 < 1.0) mub2 = 1.0;
    double Dqxqx1 = FF->xfxQ2(21, z1, mub2)/z1;
    double bWbt = 2.*M_PI * bpmag * Dqxqx1 * FF->xfxQ2(21, z2, mub2)/z2 * pdf->xfxQ2(21,xq, mub2) * exp(-1.*Sud_gggg(Q, bpmag, Subcoff));
    return bWbt;
}


int zh_kp2_b_integrated(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    double z1 = x[0] * 0.9 + 0.05;
    double z2 = x[1] * 0.9 + 0.05;
    double p1mag = helper->kTgmin;
    double thetap1 = x[2] * 2. * M_PI;
    double p2mag = helper->PhTmin;
    double y1 = x[3] * (helper->y1max - helper->y1min) + helper->y1min;
    double y2 = x[4] * (helper->y2max - helper->y2min) + helper->y2min;
    double kpmag = x[5] * helper->kpmagmax;
    double thetakp = x[6] * 2. * M_PI;
    //double xgd1 = p1mag *exp(y1)/rootsnn / (1. - p2mag*exp(y2)/rootsnn);
    /*double z1 = x[0] * (1.-xgd1) + xgd1;
    double xgd2 =  p2mag *exp(y2)/rootsnn / (1. - p1mag/z1*exp(y1)/rootsnn);
    double z2 = x[1] * (1.-xgd2) + xgd2;
    if (z1 < 0.05 || z1 > 0.95 || z2 <0.05 || z2 >0.95) {
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
        f[3] = 0.0;
        return 0;

    }
    */
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
        double Delta_phi = M_PI;
    double thetap2 = thetap1 - Delta_phi;
    double total_volume = (helper->y1max - helper->y1min) * (helper->y2max - helper->y2min)  *  2. * M_PI* 2. * M_PI *  helper->kpmagmax;// *(1.-xgd1) * (1.-xgd2);;
    double kT1 = p1mag/z1;
    double kTx1 = kT1 * cos(thetap1); 
    double kTy1 = kT1 * sin(thetap1); 
    
    double kT2 = p2mag/z2;
    double kTx2 = kT2 * cos(thetap2); 
    double kTy2 = kT2 * sin(thetap2); 
    double qTx = kTx1 + kTx2;
    double qTy = kTy1 + kTy2;
    double qT = sqrt(qTx*qTx + qTy*qTy);
    
    double kTxdiff = kpmag*cos(thetakp) - qTx;
    double kTydiff = kpmag*sin(thetakp) - qTy;
    double kminuskp_mag = sqrt(pow(kTxdiff, 2) + pow(kTydiff, 2));
    double xq = std::max(kT1, kT2) * (exp(y1) + exp(y2))/rootsnn;
    double xg = std::max(kT1, kT2) * (exp(-y1)  + exp(-y2))/rootsnn;
    //double xq = (kT1 * exp(y1) + kT2 * exp(y2))/rootsnn;
    //double xg = (kT1 * exp(-y1) + kT2 * exp(-y2))/rootsnn;
    if (xq > 1.0 || xg > 1.) {
        f[0] = 0.0;
        f[1] = 0.0;
        f[2] = 0.0;
        f[3] = 0.0;
        return 0;
    }

    double Q2 = xq * xg * rootsnn * rootsnn;
    if (Q2 <1.0) Q2 = 1.0;
    double zm1 = kT1 * exp(y1) / (kT1 * exp(y1) + kT2 * exp(y2));
    double PTtemp = pow((1.-zm1) * kTx1 - zm1 * kTx2, 2.) + pow((1.-zm1) * kTy1 - zm1 * kTy2, 2.);
    double PT1 = sqrt(PTtemp);
    //double Q2 = PTtemp; 
    double PTtemp2 = pow(zm1 * kTx1 - (1.-zm1) * kTx2, 2.) + pow(zm1 * kTy1 - (1.-zm1) * kTy2, 2.);
    double PT2 = sqrt(PTtemp2);
    
    double rapidity = log(0.01/xg);
    if (rapidity< 0.0) rapidity = 0.0;
    int y_index = int(rapidity/Y_step);
    if (kpmag > maxpT) kpmag = maxpT;
    if (kpmag < minpT) kpmag = minpT;
    
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
    double inter_F1qgVt = gsl_interp_eval(interp1, xValues.data(), F1qgVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), F2qgVt.data(), Klength);
    double inter_F2qgVt = gsl_interp_eval(interp1, xValues.data(), F2qgVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), F1ggVt.data(), Klength);
    double inter_F1ggVt = gsl_interp_eval(interp1, xValues.data(), F1ggVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), F3ggVt.data(), Klength);
    double inter_F3ggVt = gsl_interp_eval(interp1, xValues.data(), F3ggVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), FadjVt.data(), Klength);
    double inter_FadjVt = gsl_interp_eval(interp1, xValues.data(), FadjVt.data(), kpmag, nullptr);
    
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
    double inter_F1qgVt2 = gsl_interp_eval(interp2, xValues.data(), F1qgVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), F2qgVt.data(), Klength);
    double inter_F2qgVt2 = gsl_interp_eval(interp2, xValues.data(), F2qgVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), F1ggVt.data(), Klength);
    double inter_F1ggVt2 = gsl_interp_eval(interp2, xValues.data(), F1ggVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), F3ggVt.data(), Klength);
    double inter_F3ggVt2 = gsl_interp_eval(interp2, xValues.data(), F3ggVt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), FadjVt.data(), Klength);
    double inter_FadjVt2 = gsl_interp_eval(interp2, xValues.data(), FadjVt.data(), kpmag, nullptr);
    
    gsl_interp_free(interp2);
    
    inter_F1qgVt = (inter_F1qgVt2 * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F1qgVt * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_F2qgVt = (inter_F2qgVt2 * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F2qgVt * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_F1ggVt = (inter_F1ggVt2 * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F1ggVt * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_F3ggVt = (inter_F3ggVt2 * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_F3ggVt * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_FadjVt = (inter_FadjVt2 * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_FadjVt * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    if (xg>0.01) {
        double ratio = pow(1.-xg, 4.) / 0.96059601; // 0.96059601 = (1-0.01)^4
        inter_F1qgVt =  ratio * inter_F1qgVt;
        inter_F2qgVt =  ratio * inter_F2qgVt;
        inter_F1ggVt =  ratio * inter_F1ggVt;
        inter_F3ggVt =  ratio * inter_F3ggVt;
        inter_FadjVt =  ratio * inter_FadjVt;
    }
    double prefactor = p1mag * p2mag * kpmag /z1/z1/z2/z2/4./M_PI/M_PI;
    double alpha_s = pdf->alphasQ2(Q2);
    double Hqg2qg1 = alpha_s /2./pow(PT1, 4.0) * (1. + (1.-zm1)*(1.-zm1)) * (1.-zm1); // one alpha_s is canceled by F
    double Hqg2qg2 = alpha_s /2./pow(PT2, 4.0) * (1. + zm1*zm1) * zm1; // one alpha_s is canceled by F
    double Hgg2qq = alpha_s /6. / pow(PT1, 4.) * zm1*(1.-zm1) * (zm1*zm1+(1.-zm1)*(1.-zm1)); // one alpha_s is canceled by F
    double Hgg2gg = 2.*alpha_s / pow(PT1, 4.) *(zm1*zm1 + (1.-zm1) * (1.-zm1) + zm1*zm1 * (1.-zm1)*(1.-zm1)); // one alpha_s is canceled by F
    
    
    FBT ogata0 = FBT(0.0, 0, 1000); // Fourier Transform with Jnu, nu=0.0 and N=10
    
    double Subcoff = 1.0;
    double Wq2qgK1 = ogata0.fbt(std::bind(Wq2qg, std::placeholders::_1, z1, z2, xq, Q2, Subcoff), kminuskp_mag);
    double Wq2qgK2 = ogata0.fbt(std::bind(Wq2qg, std::placeholders::_1, z2, z1, xq, Q2, Subcoff), kminuskp_mag);
    double Wg2qqK = ogata0.fbt(std::bind(Wg2qq, std::placeholders::_1, z1, z2, xq, Q2, Subcoff), kminuskp_mag);
    double Wg2ggK = ogata0.fbt(std::bind(Wg2gg, std::placeholders::_1, z1, z2, xq, Q2, Subcoff), kminuskp_mag);
    
    double dsigmaqgqg_dDeltaphi = prefactor * Wq2qgK1 * ( (1.-zm1)*(1.-zm1)*inter_F1qgVt + inter_F2qgVt ) * Hqg2qg1 +
                                  prefactor * Wq2qgK2 * (zm1*zm1* inter_F1qgVt + inter_F2qgVt) * Hqg2qg2;
                                  
    double dsigmaggqq_dDeltaphi = prefactor * 2.* Wg2qqK * (inter_F1ggVt - 2.*zm1*(1.-zm1)* inter_FadjVt) * Hgg2qq;
    
    double dsigmagggg_dDeltaphi = prefactor * Wg2ggK * (inter_F1ggVt - 2.*zm1*(1.-zm1) * inter_FadjVt + inter_F3ggVt) * Hgg2gg;
    
   
    if (dsigmaqgqg_dDeltaphi < 0.0) {
        dsigmaqgqg_dDeltaphi = 0.0;
    }
    
    if (dsigmaggqq_dDeltaphi < 0.0) {
        dsigmaggqq_dDeltaphi = 0.0;
    }
    
    if (dsigmagggg_dDeltaphi < 0.0) {
        dsigmagggg_dDeltaphi = 0.0;
    }
    f[0] = (dsigmaqgqg_dDeltaphi + dsigmagggg_dDeltaphi + dsigmaggqq_dDeltaphi) * total_volume;

    f[1] = (dsigmaqgqg_dDeltaphi + dsigmagggg_dDeltaphi + dsigmaggqq_dDeltaphi) * total_volume * xg;
    
    f[2] = (dsigmaqgqg_dDeltaphi + dsigmagggg_dDeltaphi + dsigmaggqq_dDeltaphi) * total_volume * qT;
    
    return 0;
}



int main(int argc, char* argv[]) 
{
    // Set the number of threads to use
    omp_set_num_threads(num_threads);
    
    PARAMETERS params;
    int stage = std::stoi(argv[1]);
    int startid = std::stoi(argv[2]);
    int endid = std::stoi(argv[3]);
    int startid2 = std::stoi(argv[4]);
    int endid2 = std::stoi(argv[5]);
    params.etagmax    = Y1max;
    params.etagmin    = Y1min;
    params.etahmax    = Y2max;
    params.etahmin    = Y2min;
                          
    params.y1max    = Y1max;
    params.y1min    = Y1min;
    params.y2max    = Y2max;
    params.y2min    = Y2min;
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
    /* Define the integration parameters */
    const int ndim = 7; /* number of dimensions */
    const int ncomp = 3; /* number of components */
    /* Allocate memory for the results */
    cubareal integral[ncomp]; /* integral estimates */
    cubareal error[ncomp]; /* error estimates */
    cubareal prob[ncomp]; /* CHI^2 probabilities */
    
    //const int ndim2 = 3; /* number of dimensions */
//    const int ncomp2 = 1; /* number of components */
    cout << "Starts " <<endl;
    const long long int mineval = sample_points; /* minimum number of integrand evaluations */
    cout <<"step1" <<endl;
    const long long int nvec = 1; /* minimum number of integrand evaluations */
    const cubareal epsrel = 1e-5; /* relative error tolerance */
    const cubareal epsabs = 1e-5; /* absolute error tolerance */
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
//    cubareal Trigger[ncomp2]; /* integral estimates */
    //cubareal error2[ncomp2]; /* error estimates */
    //cubareal prob2[ncomp2]; /* CHI^2 probabilities */
    
    //output the results to file
    //char output_filename[128];
    std::stringstream filename;
    filename << "Xg_kT_dihadron_with_Sub_FBT_MV_" << "_"<< stage << "_" << startid<< "_" << endid 
             << "_" << startid2 << "_" << endid2 << ".txt";

    ofstream realA(filename.str());
    
        // Initialize LHAPDF
    
    LHAPDF::initPDFSet("CT18NNLO");
    pdf = LHAPDF::mkPDF("CT18NNLO", 0);
    const LHAPDF::PDFSet set("CT18NNLO"); // arxiv: 2101.04664
    // Initialize LHAPDF and set the fragmentation function set
    LHAPDF::initPDFSet("JAM20-SIDIS_FF_hadron_nlo");
    FF = LHAPDF::mkPDF("JAM20-SIDIS_FF_hadron_nlo", 0);  // Use the appropriate member index if there are multiple sets

    double detal_kTgmin; double detal_PhTmin;
    realA << "# kTgamma  PhT  dSigma_dDeltaPhi  xg*dSigma_dDeltaPhi  kT*dSigma_dDeltaPhi";
    realA << endl;
    
    if (stage == 11) {
    detal_kTgmin = 0.2; detal_PhTmin = 0.2;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2 ; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 1.;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    if (stage == 12) {
    detal_kTgmin = 0.2; detal_PhTmin = 0.5;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 1.;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin + 5.;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    if (stage == 13) {
    detal_kTgmin = 0.2; detal_PhTmin = 1.0;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 1.0;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin + 10.;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    
    if (stage == 21) {
    detal_kTgmin = 0.5; detal_PhTmin = 0.2;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2 ; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 5.;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    if (stage == 22) {
    detal_kTgmin = 0.5; detal_PhTmin = 0.5;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 5.;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin + 5.;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    if (stage == 23) {
    detal_kTgmin = 0.5; detal_PhTmin = 1.0;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 5.0;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin + 10.;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2]<< "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    if (stage == 31) {
    detal_kTgmin = 1.0; detal_PhTmin = 0.2;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2 ; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 10.;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    if (stage == 32) {
    detal_kTgmin = 1.0; detal_PhTmin = 0.5;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 10.;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin + 5.;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    if (stage == 33) {
    detal_kTgmin = 1.0; detal_PhTmin = 1.0;
    for (int ikTgmin=startid ; ikTgmin<endid; ikTgmin++) {
    for (int iPhTmin=startid2; iPhTmin<endid2; iPhTmin++) {
        params.kTgmin    = ikTgmin * 1. * detal_kTgmin + 10.0;
        params.PhTmin    = iPhTmin * 1. * detal_PhTmin + 10.;
        realA << params.kTgmin << "  " << params.PhTmin << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] << "  " << integral[2] << "  " << 1.0;
        realA << endl;
    }
    }
    }
    
    
    realA.close();
}

