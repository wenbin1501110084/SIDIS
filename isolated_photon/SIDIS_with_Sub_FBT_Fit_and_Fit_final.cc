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

const double CA = 3.;
const double CF = 4./3.;
const double Lambda2 = 0.04;// GeV^2
const double bmax2 = 2.25;//GeV^-2
const double HBARC = 0.197327053; // GeV. fm
const long long int sample_points = 1000000;
const long long int sample_points0 = 1000;
const double R_Nuclear = 6.2;//fm
const int num_threads = 10;
const double rootsnn = 200.;// GeV
const double Rcut = 0.3;
const int dipole_model = 1; //1: rcBK; 0: GBW
const double Y_step = 0.2;
const int Klength = 500;
std::vector<double> AValues, BValues, CValues, DValues, EValues, FValues, YValues_para;
//std::vector<double> KValues, NrValues;
double KValues[100000] = {0.0}; 
double NrValues[100000] = {0.0};
double YValues[100000] = {0.0};


LHAPDF::PDF* FF;
LHAPDF::PDF* pdf;

using namespace std;
/*
   llVegas is a function in the CUBA library that can be used for multidimensional numerical integration. Here is an explanation of the parameters:
    ndim: The number of dimensions of the integral.
    ncomp: The number of components of the integrand (e.g. 1 for a scalar function, 3 for a vector function, etc.).
    integrand: A pointer to the function that computes the integrand. This function must take a const int* argument specifying the length of the vector x, a const double* argument specifying the coordinates at which to evaluate the integrand, a void* argument for passing user data, and an int argument specifying the component of the integrand to evaluate. The function must return the value of the integrand at the given point.
    userdata: A pointer to any user data that should be passed to the integrand function.
    nvec: The number of vectors to be used in the Vegas integration algorithm. It should be set to zero for the default value.
    epsrel: The relative precision of the integration.
    epsabs: The absolute precision of the integration.
    flags: A bit field of flags controlling the behavior of the integrator. It should be set to zero for the default value.
    seed: The random number seed used by the integrator.
    mineval: The minimum number of function evaluations to perform.
    maxeval: The maximum number of function evaluations to perform.
    nstart: The number of samples to take in each iteration of the Vegas algorithm.
    nincrease: The number of additional samples to take if the relative variance of the integrand is too high.
    nbatch: The number of samples to take in each iteration of the Suave algorithm.
    gridno: The number of the grid file to use for the grid integration algorithm. It should be set to zero for the default value.
    statefile: The name of a file in which to save the state of the integrator. It should be set to NULL for no saving.
    spin: A pointer to any spin information that should be passed to the integrator.
    neval: A pointer to an array that will contain the number of function evaluations performed for each component of the integrand.
    fail: A pointer to an integer that will be set to a non-zero value if the integration fails.
    integral: An array that will contain the integral of each component of the integrand.
    error: An array that will contain the estimated error of each component of the integrand.
    prob: An array that will contain the estimated probability of each component of the integrand.
*/

// g++ Numerical_test.cc -lcuba -lgsl -lgslcblas -lm -fopenmp -o Diffcuba
// Parameters that need to be passed to the integrand
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
    double Delta_phi;// GeV
    double bmax;
};

extern "C"
{
double fdss_(int *is, int *ih, int *ic, int *io, double *x, double *Q2, double *u, double *ub, double *d, double *db, double *s, double *sb, double *c, double *b, double *gl);
}


int S_Sub(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    // x0: mu2
    double bstar = helper->b / sqrt(1. + helper->b * helper->b / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    if (mub > sqrt(helper->Q2)) mub= sqrt(helper->Q2);
    double mu2 = x[0] * (helper->Q2 - mub*mub) + mub*mub;
    //double beta0 = 0.75;//(11. - 2.) / 12.;
    //double as = 4.*M_PI /(0.75 * log(mu2/Lambda2) + 50.);//50: regulator
    //double as = 0.118;
    double alpha_s = pdf->alphasQ2(mu2);// call from LHAPDF
    double A = alpha_s / M_PI * (CF + CA/2.);
    double B = -1.5*alpha_s /M_PI *CF;
    f[0] = (helper->Q2 - mub*mub) * (A*log( helper->Q2 / mu2 ) + B) / mu2;
    return 0;
}

int S_Sub_Guass_Leg(double mu2, double Q2){
    // x0: mu2
    //double beta0 = 0.75;//(11. - 2.) / 12.;
    //double as = 4.*M_PI /(0.75 * log(mu2/Lambda2) + 50.);//50: regulator
    
    double alpha_s = pdf->alphasQ2(mu2);// call from LHAPDF
    double A = alpha_s / M_PI * (CF + CA/2.);
    double B = -1.5*alpha_s /M_PI *CF;
    return (A*log( Q2 / mu2 ) + B) / mu2;
}

// Define the Legendre polynomial and its derivative
double legendrePoly(int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    return ((2.0 * n - 1.0) * x * legendrePoly(n - 1, x) - (n - 1) * legendrePoly(n - 2, x)) / n;
}


double legendrePolyDerivative(int n, double x) {
    return (x * legendrePoly(n, x) - legendrePoly(n - 1, x)) / (x * x - 1);
}

// Function to compute the Legendre roots using Newton-Raphson
void computeLegendreRoots(int n, std::vector<double>& x, std::vector<double>& w) {
    const double tolerance = 1e-15;
    for (int i = 0; i < n; ++i) {
        double x0 = cos(M_PI * (i + 0.75) / (n + 0.5));  // Initial guess using Chebyshev nodes
        double x1 = 0.0;
        while (std::abs(x1 - x0) > tolerance) {
            x1 = x0 - legendrePoly(n, x0) / legendrePolyDerivative(n, x0);
            x0 = x1;
        }
        x[i] = x1;
        w[i] = 2.0 / ((1.0 - x1 * x1) * legendrePolyDerivative(n, x1) * legendrePolyDerivative(n, x1));
    }
}

// Gauss-Legendre quadrature function
double gaussLegendre(int n, double a, double b, double Q2) {
    std::vector<double> x(n);  // Quadrature points
    std::vector<double> w(n);  // Quadrature weights

    computeLegendreRoots(n, x, w);

    double integral = 0.0;
    double total_w = 0.0;
    for (int i = 0; i < n; ++i) {
        double xi = 0.5 * ((b - a) * x[i] + a + b);
        integral += w[i] * S_Sub_Guass_Leg(xi, Q2);
        total_w = total_w + w[i];
    }

    return  (b - a) * integral/total_w;
}




/* Define the W(b) function, use the FBT method */
double Wb(double b, double zh, double xp, double Q2){
    
    //int n = 5;  // Number of quadrature points
    double bstar = b / sqrt(1. + b * b / bmax2);
    double mub = 1.1224972160321824/bstar;//1.1224972160321824 = 2exp(-gammaE)
    // Evaluate the fragmentation function at the specified parameters
    double mub2 = mub * mub;
    if (mub2 > Q2) mub2= Q2;
    if (mub2 < 1.0) mub2 = 1.0;

    //double SSub = gaussLegendre(n, mub2, Q2, Q2);
    double beta = 0.776305; 
    double Lambda2 = 0.0542;//GeV^2  to fit the JAM20 PDF
    double A = 0.899;
    double B = -0.631;// A and B are values of the Sudakov only
    // A = 0.899, B =-0.631 for B = 2Bq; 
    // A = 0.897, B = -1.341 for B = 2Bq +Bg;
    double SSub = 1./ beta * ( -1.*A *log(Q2/mub2) + (B + A*log(Q2/Lambda2)) * log( (log(Q2/Lambda2)) / (log(mub2/Lambda2)) ) );
    double S_nonpert = 0.212 * b*b + 0.21*log(Q2 / 2.4) * log(1. + b * b/bmax2);
    
    /*
    // Fragmentation functon
    int is=1, ih=4, ic=0, io=1;
    double u, ub, d, db, s, sb, c, bb, gl;
    fdss_(&is, &ih, &ic, &io, &zh, &mub2, &u, &ub, &d, &db, &s, &sb, &c, &bb, &gl);
    */
    double bWbt = b * 0.02418865082489962 *exp(-SSub - S_nonpert) * (8./9. * FF->xfxQ2(2, zh, mub2)/zh * pdf->xfxQ2(2,xp, mub2)/xp +
                  1./9. * FF->xfxQ2(1, zh, mub2)/zh * pdf->xfxQ2(1,xp, mub2)/xp ); 
    //double bWbt = b * 0.02418865082489962 * exp(-SSub - S_nonpert) * (8./9. * u/zh * pdf->xfxQ2(2,xp, mub2)/xp +
    //              1./9. * d/zh * pdf->xfxQ2(1,xp, mub2)/xp ); 
    //double bWbt = b * 0.02418865082489962 * (8./9. * FF->xfxQ2(2, zh, mub2) * pdf->xfxQ2(2,xp, mub2) +
    //              1./9. * FF->xfxQ2(1, zh, mub2) * pdf->xfxQ2(1,xp, mub2) ); 
                 // 0.0038497433455066264 = 3/8/pi**4; 0.02418865082489962 = 3/8/pi**4 * 2 * pi
    return bWbt;
}



int zh_kp2_b_integrated(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    // x0: zh; x1: kp; x2: thetakp; x3 etagamma; x4: etah; x5: kTgamma; x6: PhT; x7: theta_kT
    double zh = x[0] * 0.9 + 0.05;
    double kpmag = x[1] * helper->kpmagmax;
    double thetakp = x[2] * 2. * M_PI;
    double etag = x[3] * (helper->etagmax - helper->etagmin) + helper->etagmin;
    double etah = x[4] * (helper->etahmax - helper->etahmin) + helper->etahmin;
    double kTg  = x[5] * (helper->kTgmax - helper->kTgmin)   + helper->kTgmin;
    double PhT  = x[6] * (helper->PhTmax - helper->PhTmin)   + helper->PhTmin;
    double R2condition = (etag - etah) * (etag - etah) + helper->Delta_phi * helper->Delta_phi;
    if (R2condition < Rcut*Rcut) {
        f[0] = 0.0;
        return 0;
    }
    double theta_kT = x[7] * 2. * M_PI;
    //double bmag = x[8] * helper->bmax;
    double theta_PhT = theta_kT - helper->Delta_phi;
    double total_volume = (helper->etagmax - helper->etagmin) * (helper->etahmax - helper->etahmin) * (helper->kTgmax - helper->kTgmin) *
                          (helper->PhTmax - helper->PhTmin) *  2. * M_PI * 2. * M_PI * helper->kpmagmax;
    // sigma^hat
    double z = exp(etag) * kTg / (exp(etag)* kTg + exp(etah) * PhT / zh);
    double qdotk = 0.5 * kTg * kTg / z/(1.-z); // back-to-back limit
    //double qdotk = kTg * PhT / zh * (0.5 * exp(etag - etah) + 0.5 * exp(etah - etag) - cos(helper->Delta_phi));
    double sigmahat = 0.0012165450121654502 * (1. + (1.-z)*(1.-z)) * z / qdotk /kTg/kTg; // 0.0012165450121654502 = 1/137/6
    double kTxdiff = kTg*cos(theta_kT) + PhT * cos(theta_PhT) / zh - kpmag*cos(thetakp);
    double kTydiff = kTg*sin(theta_kT) + PhT * sin(theta_PhT) / zh - kpmag*sin(thetakp);
    double kminuskp_mag = sqrt(pow(kTxdiff, 2) + pow(kTydiff, 2));
    double xp = std::max(kTg, PhT/zh) * (exp(etag) + exp(etah))/rootsnn;
    double xg = std::max(kTg, PhT/zh) * (exp(-etag)  + exp(-etah))/rootsnn;
    //double xp = (exp(etag) * kTg + exp(etah) * PhT/zh ) / rootsnn;
    //double xg = (exp(-etag) * kTg + exp(-etah) * PhT/zh ) / rootsnn;
    //if (Q2 > 1.e5) Q2  = 9.999e4;
    if (xp > 1.0 || xg > 1.) {
        f[0] = 0.0;
        return 0;
    }
    // Use the integration FBT
    double Q2 = rootsnn * rootsnn * xp * xg;
    if (Q2 < 1.0 ) Q2 = 1.0; 
    FBT ogata0 = FBT(0.0, 0, 3); // Fourier Transform with Jnu, nu=0.0 and N=10
    double Wk = ogata0.fbt(std::bind(Wb, std::placeholders::_1, zh, xp, Q2), kminuskp_mag);
    //double Wk = Wb(bmag, zh, xp, Q2) *  boost::math::cyl_bessel_j(0, bmag *kminuskp_mag );
    // N tidle // GBW, photon-nucleon
    //double G = 0.25 * pow(0.0003/xg, 0.29); 
    double Ntidle = 0.0;
    if (xg > 0.01) { // Using the Match
        double ap = pdf->xfxQ2(21, xg, 6.) / (pdf->xfxQ2(21, 0.01, 6.));
        double rapidity = log(0.01/0.01);
        if (rapidity > 15.8) rapidity = 15.8;
        double Ntidle0 = 0.0;
        if (dipole_model == 0) {
            double G = 0.25 * pow(0.01/0.01, 0.29) * exp(0.29 * rapidity); 
            Ntidle0 = exp(-kpmag*kpmag/G*0.25)/G*0.5; // ????
        }
        if (dipole_model == 1) {
            std::vector<double> xValues, yValues;
            xValues.clear(); yValues.clear();
            for (int inn= 0; inn <  Klength; inn++) {
                xValues.push_back(KValues[inn]);
                yValues.push_back(NrValues[inn]);
            }
            if (kpmag < 2.) {
                gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
                gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
                Ntidle0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), kpmag, nullptr);
                gsl_interp_free(interp);
            }
            if (kpmag >= 2.) {
                Ntidle0 = AValues[0] * ( -1. * pow(kpmag, BValues[0]) * CValues[0] + DValues[0]) +
                          EValues[0] * pow(kpmag, FValues[0]);
                Ntidle0 = exp(Ntidle0); // Fit the log(data)
            }
        }
        Ntidle = ap * Ntidle0;
    } else {
        double rapidity = log(0.01/xg);
        if (rapidity > 15.0) rapidity = 15.;
        if (dipole_model == 0) {
            double G = 0.25 * pow(0.01/xg, 0.29) * exp(0.29 * rapidity); 
            Ntidle = exp(-kpmag*kpmag/G*0.25)/G*0.5; // ????
        } 
        if (dipole_model == 1) {
            int y_index = int(rapidity/Y_step);
            double Ntidle1 = 0;
            double Ntidle2 = 0;
            if ( kpmag < 2.) {
                double xValues3[Klength], yValues3[Klength];
                for (int inn= y_index * Klength; inn <  y_index * Klength + Klength; inn++) {
                    xValues3[inn] = (KValues[inn]);
                    yValues3[inn] = (NrValues[inn]);
                }
            
                gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
                gsl_interp_init(interp, xValues3, yValues3, Klength);
                Ntidle1 = gsl_interp_eval(interp, xValues3, yValues3, kpmag, nullptr);
                gsl_interp_free(interp);
                /////
                int y_index2 = y_index +1;
                //std::vector<double> xValues2, yValues2;
                //xValues2.clear(); yValues2.clear();
                double xValues2[Klength], yValues2[Klength];
                for (int inn= y_index2 * Klength; inn <  y_index2 * Klength + Klength; inn++) {
                    xValues2[inn] =(KValues[inn]);
                    yValues2[inn] =(NrValues[inn]);
                }
                gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Klength);
                gsl_interp_init(interp2, xValues2, yValues2, Klength);
                Ntidle2 = gsl_interp_eval(interp2, xValues2, yValues2, kpmag, nullptr);
                gsl_interp_free(interp2);
            }
            if ( kpmag >= 2.) {
                Ntidle1 = AValues[y_index] * ( -1. * pow(kpmag, BValues[y_index]) * CValues[y_index] + DValues[y_index]) +
                          EValues[y_index] * pow(kpmag, FValues[y_index]);
                Ntidle2 = AValues[y_index+1] * ( -1. * pow(kpmag, BValues[y_index+1]) * CValues[y_index+1] + DValues[y_index+1]) +
                          EValues[y_index+1] * pow(kpmag, FValues[y_index+1]); 
                Ntidle1 = exp(Ntidle1); // Fit the log(data)
                Ntidle2 = exp(Ntidle2); // Fit the log(data)
            }
            Ntidle = (Ntidle1 * (rapidity - YValues_para[y_index] ) + 
                Ntidle2 * (YValues_para[y_index+1] - rapidity) ) / Y_step;
        }
    }
    
    // integrate the whole function
    double  dsigma_dDeltaphi = 0.025330295910584444 * M_PI * R_Nuclear * R_Nuclear / zh/zh * Wk * pow(kpmag, 3) * Ntidle * sigmahat *
                              kTg * PhT; // Jacobe
                              //0.025330295910584444 = 1/(2pi)^2
    if (dsigma_dDeltaphi < 0.0) {
        dsigma_dDeltaphi = 0.0;
        cout << "ddddddddd " << dsigma_dDeltaphi << "  " <<  Wk << "  " << sigmahat << "   " << endl;
    }
    f[0] = dsigma_dDeltaphi * total_volume;
    return 0;
}



int main(int argc, char* argv[]) {

    // Set the number of threads to use
    omp_set_num_threads(num_threads);
    
    PARAMETERS params;
    int startid = std::stoi(argv[1]);
    int endid = std::stoi(argv[2]);
    int stage = std::stoi(argv[3]);
    double kgTmin = std::stod(argv[4]);
    double kgTmax = std::stod(argv[5]);
    double phTmin = std::stod(argv[6]);
    double phTmax = std::stod(argv[7]); // Convert the 5th command-line argument to a double



    params.etagmax    = 0.35;
    params.etagmin    = -0.35;
    params.kpmagmax    = 20.0;///////////////////
    params.etahmax    = 0.35;
    params.etahmin    = -0.35;
    params.kTgmax    = kgTmax;
    params.kTgmin    = kgTmin;
    params.PhTmax    = phTmax;
    params.PhTmin    = phTmin;
    params.bmax      = 5.;// GeV^-1

    // Read in Fitted parameters
    std::ifstream inputFile("All_fit_paras_log_2_5_final_fit_rcBK_MV_Heikki_table");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Read data from the file
    double Ymid, Amid, Bmid, Cmid, Dmid, Emid, Fmid; // r in [GeV^-1]
    while (inputFile >> Ymid >> Amid >> Bmid >> Cmid >> Dmid >> Emid >> Fmid) {
        YValues_para.push_back(Ymid);
        AValues.push_back(Amid);
        BValues.push_back(Bmid);
        CValues.push_back(Cmid);
        DValues.push_back(Dmid);
        EValues.push_back(Emid);
        FValues.push_back(Fmid);
    }
    // Close the file
    inputFile.close();
    
    std::ifstream inputFile2("rcNK_table_K_all_rcBK_MVe_Heikki_table");
    if (!inputFile2.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Read data from the file
    double Y, KK, Nr; // r in [GeV^-1]
    int ikik = 0;
    while (inputFile2 >> Y >> KK >> Nr) {
        //YValues.push_back(Y);
        //KValues.push_back(KK);
        //NrValues.push_back(Nr);
        YValues[ikik] = Y;
        KValues[ikik] = KK; 
       // cout << "dffffffffffff" << KValues[ikik] << "   " << ikik << endl;
        NrValues[ikik] = Nr;
        ikik++;
    }
    // Close the file
    inputFile2.close();
    
    
    /* Define the integration parameters */
    const int ndim = 8; /* number of dimensions */
    const int ncomp = 1; /* number of components */
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
    cubareal integral[ncomp]; /* integral estimates */
    cubareal error[ncomp]; /* error estimates */
    cubareal prob[ncomp]; /* CHI^2 probabilities */
    

    
    //output the results to file
    //char output_filename[128];
    std::stringstream filename;
    filename << "dSigma_dDeltaPhi_with_Sub_FBT_" << startid << "_" << endid << "_" <<  stage 
             << "_"<< kgTmin << "_" << kgTmax << "_" << phTmin << "_" << phTmax <<".txt";

    //sprintf(output_filename,"dSigma_dDeltaPhi_with_Sub_FBT");
    //std::ofstream outputFile(filename.str());
    ofstream realA(filename.str());
    
        // Initialize LHAPDF
    LHAPDF::initPDFSet("JAM20-SIDIS_PDF_proton_nlo");
    // Access PDF set
    pdf = LHAPDF::mkPDF("JAM20-SIDIS_PDF_proton_nlo", 0);
    const LHAPDF::PDFSet set("JAM20-SIDIS_PDF_proton_nlo"); // arxiv: 2101.04664

    // Initialize LHAPDF and set the fragmentation function set
    LHAPDF::initPDFSet("JAM20-SIDIS_FF_hadron_nlo");
    FF = LHAPDF::mkPDF("JAM20-SIDIS_FF_hadron_nlo", 0);  // Use the appropriate member index if there are multiple sets

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
                /* Print the results */
                //printf("Integral estimate: %e\n", integral[0]);
                //printf("Error estimate: %e\n", error[0]);
                //printf("Number of evaluations: %lld\n", neval);
                //printf("Status flag: %d\n", fail);
                realA << integral[0] <<  "  ";
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
                /* Print the results */
                //printf("Integral estimate: %e\n", integral[0]);
                //printf("Error estimate: %e\n", error[0]);
                //printf("Number of evaluations: %lld\n", neval);
                //printf("Status flag: %d\n", fail);
                realA << integral[0] <<  "  ";
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
                /* Print the results */
                //printf("Integral estimate: %e\n", integral[0]);
                //printf("Error estimate: %e\n", error[0]);
                //printf("Number of evaluations: %lld\n", neval);
                //printf("Status flag: %d\n", fail);
                realA << integral[0] <<  "  ";
        realA << endl;
    }
    }   
    
    realA.close();
    return 0;
}
