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

std::vector<double> YValues, pTValues, FWWV, HWWV;
const int Klength = 700;
const double minpT = 1.e-05;
const double maxpT = 99.;
const double Y_step = 0.2;
const double ST = 1.0;//M_PI * 6.6*6.6 /0.197/0.197;// GeV^-2
const double CA = 3.;
const double CF = 4./3.;
const double alpha_em = 1./137.;
const double betaSud = 0.68714457;
const double Lambda2Sud = 0.03006785;
double Lambda = sqrt(Lambda2Sud);
const double bmax2 = 0.25;//GeV^-2
const double Q0 = 1.5491933384829668;//GeV
const double g1 = 0.212;
const double g2 = 0.84;
const double rootsnn = 200.;// GeV
const double Y1max = 2.;
const double Y1min = 1.;
const double Y2max = 2.;
const double Y2min = 1.;
const double Q2minIn = 5.0;// GeV^2
const double Q2maxIn = 5.0;// GeV^2
const double Rcut = 0.4;
const int num_threads = 10;
const long long int sample_points = 2000;

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
    double Q2max;
    double Q2min;
};

using namespace std;

double Sud_gammagqqbar(double PT, double bmag, double z, double Q2, double Subcoff) {
    double bstar = bmag / sqrt(1. + bmag*bmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    if (mub > PT) mub = PT;
    double A1 =  9.01878614e-01; double B1 = -1.43510990e-06; 
    double A2 =  2.74951772e-05; double B2 = 1.59052135e-01;
    double Sudp1 = 1./betaSud * (-2.*A1*log(PT/mub) + (B1 + 2.*A1*log(PT/Lambda)) * (log((log(PT/Lambda)) / (log(mub/Lambda)))));
    double Sudp2 = 1./betaSud * (-2.*A2*log(PT/mub) + (B2 + 2.*A2*log(PT/Lambda)) * (log((log(PT/Lambda)) / (log(mub/Lambda)))));
    double Qbar2 = z*(1.-z) * Q2;
    double PT2 = PT * PT;
    Sudp2 = Sudp2 * ( log( 1.+ Qbar2/PT2) - 3.*CF - CF/3.*log(z*(1.-z)) );
    double Sudnp = g1*bmag *bmag + g2 *log(PT/Q0) * log(bmag/bstar);
    double Sud = Sudp1 + Sudp2 + Subcoff * Sudnp;
    //if (Sud <0.) Sud = 0.;
    return Sud;
}

double Wgammag2qqbar(double bpmag, double z1, double z2, double z, double PT, double Q2, double Subcoff) {
    double bstar = bpmag / sqrt(1. + bpmag*bpmag / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    double mub2 = mub * mub;
    if (std::isnan(mub2)) {
        mub2 = Q2;
        std::cout << "Warning: myValue is NaN " << bstar << "  " << bpmag << std::endl;
    } 
    if (mub2 > Q2) mub2= Q2;
    //if (mub2 < 1.) return 0.0;
    double Dqxqx1 = 4./9. * FF->xfxQ2(2, z1, mub2)/z1 * FF->xfxQ2(2, z2, mub2)/z2  + 
                    1./9. * FF->xfxQ2(1, z1, mub2)/z1 * FF->xfxQ2(1, z2, mub2)/z2 +
                    1./9. * FF->xfxQ2(3, z1, mub2)/z1 * FF->xfxQ2(3, z2, mub2)/z2;
    double bWbt = 2.*M_PI * bpmag * Dqxqx1 * exp(-1.*Sud_gammagqqbar(PT, bpmag, z, Q2, Subcoff));
    //if (bWbt < 0.) bWbt = 0.;
    return bWbt;
}

int zh_kp2_b_integrated(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    double z1 = x[0] * 0.9 + 0.05;
    double z2 = x[1] * 0.9 + 0.05;
    double p1mag = x[2] * (helper->p1magmax - helper->p1magmin) + helper->p1magmin;
    double thetap1 = x[3] * 2. * M_PI;
    double pt2maxtep = helper->p2magmax;// std::min(helper->p2magmax, p1mag);
    double p2mag = x[4] * (pt2maxtep - helper->p2magmin) + helper->p2magmin;
    double y1 = x[5] * (helper->y1max - helper->y1min) + helper->y1min;
    double y2 = x[6] * (helper->y2max - helper->y2min) + helper->y2min;
    double kpmag = x[7] * helper->kpmagmax;
    double thetakp = x[8] * 2. * M_PI;
    double kxp = kpmag*cos(thetakp);
    double kyp = kpmag*sin(thetakp);
    double Q2 = helper->Q2min;//x[9] * (helper->Q2max - helper->Q2min) + helper->Q2min;
    double kT1 = p1mag/z1;
    double kT2 = p2mag/z2;
    double y_inelasticity_root = (kT1 * exp(y1) + kT2 * exp(y2) )/ rootsnn;
    double y_inelasticity = y_inelasticity_root * y_inelasticity_root;
    //cout << "y_inelasticity = " << y_inelasticity << endl;
    if (y_inelasticity >0.95 || y_inelasticity < 0.01) { // experimental cut
        f[0] = 0.0;
        f[1] = 0.0;
        return 0;
    }
    
    double thetap2 = thetap1 - helper->Delta_phi;
    double total_volume = (helper->y1max - helper->y1min) * (helper->y2max - helper->y2min) * (helper->p1magmax - helper->p1magmin) *
                          (pt2maxtep - helper->p2magmin) *  2. * M_PI* 2. * M_PI *  helper->kpmagmax * 0.9*0.9;
    
    double kTx1 = kT1 * cos(thetap1); 
    double kTy1 = kT1 * sin(thetap1); 
    
    double kTx2 = kT2 * cos(thetap2); 
    double kTy2 = kT2 * sin(thetap2); 
    double qTx = kTx1 + kTx2;
    double qTy = kTy1 + kTy2;
    //double qT = sqrt(qTx*qTx + qTy*qTy);
    
    double zm1 = kT1 * exp(y1) / (kT1 * exp(y1) + kT2 * exp(y2));
    double PX = (1.-zm1) * kTx1 - zm1 * kTx2; double PY = (1.-zm1) * kTy1 - zm1 * kTy2;
    double PT2 = PX * PX + PY * PY;
    double PT = sqrt(PT2);
    double cosphiprime = (PX * kxp + PY * kyp) / PT / kpmag;
    double cos2phiprime = 2. * cosphiprime *  cosphiprime -1.;
    double Qbar2 = zm1 * (1.-zm1) * Q2;
    
    double kTxdiff = kxp - qTx;
    double kTydiff = kyp - qTy;
    double kminuskp_mag = sqrt(pow(kTxdiff, 2) + pow(kTydiff, 2));
    double xg = Q2 / y_inelasticity /rootsnn/rootsnn + (kT1 * exp(-y1)  + kT2 * exp(-y2))/rootsnn/y_inelasticity_root;
    //cout << xg << endl;
    if (xg > 1.) {
        f[0] = 0.0;
        f[1] = 0.0;
        return 0;
    }
    double prefactor = alpha_em*alpha_em/(M_PI * Q2 * pow(PT2 + Qbar2, 4.));
    double Heg2eqqbar = prefactor * ( (1.-y_inelasticity) * 8. *zm1 *zm1 *(1.-zm1) *(1.-zm1) *PT2*Qbar2 +
           0.5*(1.+(1.-y_inelasticity)*(1.-y_inelasticity)) * zm1 *(1.-zm1) * (zm1*zm1 + (1.-zm1) *(1.-zm1)) * (PT2*PT2 + Qbar2 * Qbar2) 
                       );
    double H2eg2eqqbar = prefactor * ( (1.-y_inelasticity) * 8. * zm1*zm1*(1.-zm1) *(1.-zm1) *PT2*Qbar2 - 
                                       (1.+(1.-y_inelasticity)*(1.-y_inelasticity) ) * Qbar2 * PT2
                        );
    
    double rapidity = log(0.01/xg);
    if (rapidity< 0.0) rapidity = 0.0;
    int y_index = int(rapidity/Y_step);
    if (kpmag > maxpT) kpmag = maxpT;
    if (kpmag < minpT) kpmag = minpT;
    
    std::vector<double> xValues, FWWt, HWWt;
    xValues.clear(); FWWt.clear(); HWWt.clear();
    for (int inn= y_index * Klength; inn < y_index * Klength + Klength; inn++) {
        xValues.push_back(pTValues[inn]);
        FWWt.push_back(FWWV[inn]);
        HWWt.push_back(HWWV[inn]);
    }
    
    gsl_interp *interp1 = gsl_interp_alloc(gsl_interp_cspline, Klength);
    
    gsl_interp_init(interp1, xValues.data(), FWWt.data(), Klength);
    double inter_FWWt = gsl_interp_eval(interp1, xValues.data(), FWWt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp1, xValues.data(), HWWt.data(), Klength);
    double inter_HWWt = gsl_interp_eval(interp1, xValues.data(), HWWt.data(), kpmag, nullptr);
    
    gsl_interp_free(interp1);
    
    xValues.clear(); FWWt.clear(); HWWt.clear();
    y_index = y_index +1;
    for (int inn= y_index * Klength; inn < y_index * Klength + Klength; inn++) {
        xValues.push_back(pTValues[inn]);
        FWWt.push_back(FWWV[inn]);
        HWWt.push_back(HWWV[inn]);
    }
    
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_cspline, Klength);
    
    gsl_interp_init(interp2, xValues.data(), FWWt.data(), Klength);
    double inter_FWWt2 = gsl_interp_eval(interp2, xValues.data(), FWWt.data(), kpmag, nullptr);
    
    gsl_interp_init(interp2, xValues.data(), HWWt.data(), Klength);
    double inter_HWWt2 = gsl_interp_eval(interp2, xValues.data(), HWWt.data(), kpmag, nullptr);
    
    gsl_interp_free(interp2);
    
    inter_FWWt = (inter_FWWt2 * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_FWWt * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    inter_HWWt = (inter_HWWt2 * (rapidity - YValues[y_index*Klength-Klength] ) + 
                    inter_HWWt * (YValues[y_index*Klength] - rapidity) ) / Y_step;
    if (xg>0.01) {
        double ratio = pow(1.-xg, 4.) / 0.96059601; // 0.96059601 = (1-0.01)^4
        inter_FWWt =  ratio * inter_FWWt;
        inter_HWWt =  ratio * inter_HWWt;
    }
        
    FBT ogata0 = FBT(0.0, 0, 1000); // Fourier Transform with Jnu, nu=0.0 and N=10
    double Subcoff = 1.0;
    double Wgammag2qqbarK = ogata0.fbt(std::bind(Wgammag2qqbar, std::placeholders::_1, z1, z2, zm1, PT, Q2, Subcoff), kminuskp_mag);
    double prefactor_DIS = p1mag * p2mag /z1/z1/z2/z2 /4./M_PI/M_PI * kpmag * Wgammag2qqbarK * total_volume;
    double dsigmaqgqg_dDeltaphi =  prefactor_DIS * inter_FWWt * Heg2eqqbar;
    double dsigmaggqq_dDeltaphi =  prefactor_DIS * inter_HWWt * H2eg2eqqbar * cos2phiprime;
    
   
    if (dsigmaqgqg_dDeltaphi < 0.0) {
        dsigmaqgqg_dDeltaphi = 0.0;
    }
    
    if (dsigmaggqq_dDeltaphi < 0.0) {
        dsigmaggqq_dDeltaphi = 0.0;
    }
    
    f[0] = dsigmaqgqg_dDeltaphi;
    f[1] = dsigmaggqq_dDeltaphi;
    //cout << dsigmaqgqg_dDeltaphi << "  " << dsigmaggqq_dDeltaphi << endl;
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
    params.Q2min       = Q2minIn;
    params.Q2max       = Q2maxIn;
    // Open the input file
    std::ifstream inputFile("Paul_table/Regularged_FWW_proton_MV_Paul");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Read data from the file
    double Y, pTm, F1qgm, F2qgm;
    while (inputFile >> Y >> pTm >> F1qgm >> F2qgm) {
        YValues.push_back(Y);
        pTValues.push_back(pTm);
        FWWV.push_back(F1qgm);
        HWWV.push_back(F2qgm);
    }
    inputFile.close();
    
        /* Define the integration parameters */
    //const int ndim2 = 3; /* number of dimensions */
    const int ncomp2 = 1; /* number of components */
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
    cubareal Trigger[ncomp2]; /* integral estimates */
    //cubareal error2[ncomp2]; /* error estimates */
    //cubareal prob2[ncomp2]; /* CHI^2 probabilities */
    std::stringstream filename;
    filename << "dSigma_dDeltaPhi_dihadron_DIS_" << startid << "_" << endid << "_" <<  stage 
             << "_"<< P1magmin << "_" << P1magmax << "_" << P2magmin << "_" << P2magmax <<".txt";
    ofstream realA(filename.str());
    
    LHAPDF::initPDFSet("CT18NNLO");
    pdf = LHAPDF::mkPDF("CT18NNLO", 0);
    const LHAPDF::PDFSet set("CT18NNLO"); // arxiv: 2101.04664
    LHAPDF::initPDFSet("JAM20-SIDIS_FF_hadron_nlo");
    FF = LHAPDF::mkPDF("JAM20-SIDIS_FF_hadron_nlo", 0);  // Use the appropriate member index if there are multiple sets
   /* 
    llVegas(ndim2, ncomp2, Calculate_trigger, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, Trigger, error2, prob2);
    cout << Trigger[0] << endl;
   */
   Trigger[0] = 1.0;
    /* Define the integration parameters */
    const int ndim = 9; /* number of dimensions */
    const int ncomp = 2; /* number of components */
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
                realA << integral[0] <<  "  " << integral[1] <<  "  "<<  "  " <<  "  " << Trigger[0];
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
                realA << integral[0] <<  "  " << integral[1] <<  "  "<<  "  " <<   "  "  << Trigger[0];
        realA << endl;
    }
    }
    if (stage == 2) {
    detal_theta = 2.0; //params.Delta_phi;
    length = 60;
    detal_theta_step = (M_PI - detal_theta) / length;
    for (int itheta=1+ startid; itheta<endid +1; itheta++) {
        params.Delta_phi = itheta * 1. * detal_theta_step + detal_theta; 
        realA << params.Delta_phi << "  ";
                /* Call the integrator */
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                realA << integral[0] <<  "  " << integral[1] <<  "  "<<  "  " <<  "  " << Trigger[0];
        realA << endl;
    }
    }   
    realA.close();
}

