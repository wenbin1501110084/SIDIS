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

const double CA = 3.;
const double CF = 4./3.;
const double Lambda2 = 0.04;// GeV^2
const double bmax2 = 2.25;//GeV^-2
const double HBARC = 0.197327053; // GeV. fm
const long long int sample_points = 100000;
const long long int sample_points0 = 10000;
const double R_Nuclear = 6.2;//fm
const int num_threads = 6;
const double Y1max = 0.35;
const double Y1min = -0.35;
const double Y2max = 0.35;
const double Y2min = -0.35;
const double Rcut = 0.4;
const double rootsnn = 200.;
const int dipole_model = 1; //1: rcBK; 0: GBW
const double Y_step = 0.2;
const int Klength = 700;
const double minpT = 1.e-05;
const double maxpT = 99.;

std::vector<double> YValues, pTValues, F1qgV, F2qgV, F1ggV, F3ggV, F6ggV, FadjV;
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
    double Rp2;
    double x0;
};

extern "C"
{
double fdss_(int *is, int *ih, int *ic, int *io, double *x, double *Q2, double *u, double *ub, double *d, double *db, double *s, double *sb, double *c, double *b, double *gl);
}


double interp_r( double r, std::vector<double>& xValues, std::vector<double>& yValues) { 
    const size_t numPoints = 500;
    // Create a linear interpolation object
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, numPoints);
    gsl_interp_init(interp, xValues.data(), yValues.data(), numPoints);
    return gsl_interp_eval(interp, xValues.data(), yValues.data(), r, nullptr);
}

// Function for linear interpolation
double linearInterpolation(double x, const double xData[], const double yData[], int dataSize) {
    for (int i = 0; i < dataSize - 1; ++i) {
        if (x >= xData[i] && x <= xData[i + 1]) {
            // Calculate the slope (m) of the line passing through (x1, y1) and (x2, y2)
            double m = (yData[i + 1] - yData[i]) / (xData[i + 1] - xData[i]);

            // Use the equation of a line to interpolate the y-value for the given x
            double y = yData[i] + m * (x - xData[i]);
            return y;
        }
    }

    // If x is outside the range of the data points, return NaN (Not a Number)
    return std::numeric_limits<double>::quiet_NaN();
}


int zh_kp2_b_integrated(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    // x0: zh; x1: kp; x2: thetakp; x3 etagamma; x4: etah; x5: kTgamma; x6: PhT; x7: theta_kT
    double zh = x[0] * 0.9 + 0.05 ;
    double etag = x[1] * (helper->etagmax - helper->etagmin) + helper->etagmin;
    double etah = x[2] * (helper->etahmax - helper->etahmin) + helper->etahmin;
    double kTg  = x[3] * (helper->kTgmax - helper->kTgmin)   + helper->kTgmin;
    double PhT  = x[4] * (helper->PhTmax - helper->PhTmin)   + helper->PhTmin;
    double R2condition = (etag - etah) * (etag - etah) + helper->Delta_phi * helper->Delta_phi;
    if (R2condition < Rcut*Rcut) {
        f[0] = 0.0;
        return 0;
    }
    double xp = std::max(kTg, PhT/zh) * (exp(etag) + exp(etah))/rootsnn;
    double xg = std::max(kTg, PhT/zh) * (exp(-etag)  + exp(-etah))/rootsnn;
    double Q2 = rootsnn * rootsnn * xp * xg;
    //if (Q2 > 1.e5) Q2  = 9.999e4;
    if (Q2 < 1.0 ) Q2 = 1.0; 
    if (xp > 1.0 || xg > 1.) {
        f[0] = 0.0;
        return 0;
    }
    
    double theta_kT = x[5] * 2. * M_PI;
    double theta_PhT = theta_kT - helper->Delta_phi;
    double total_volume = (helper->etagmax - helper->etagmin) * (helper->etahmax - helper->etahmin) * (helper->kTgmax - helper->kTgmin) *
                          (helper->PhTmax - helper->PhTmin) *  2. * M_PI * 1.0;
    // sigma^hat
    double z = exp(etag) * kTg / (exp(etag)* kTg + exp(etah) * PhT / zh);
    double qdotk = 0.5 * kTg * kTg / z/(1.-z);
    double sigmahat = 0.0012165450121654502 * (1. + (1.-z)*(1.-z)) * z / qdotk /kTg/kTg; // 0.0012165450121654502 = 1/137/6
    
    //double xp = (exp(etag) * kTg + exp(etah) * PhT/zh ) / rootsnn;
    //double xg = (exp(-etag) * kTg + exp(-etah) * PhT/zh ) / rootsnn;
    double kTxdiff = kTg*cos(theta_kT) + PhT * cos(theta_PhT) / zh;
    double kTydiff = kTg*sin(theta_kT) + PhT * sin(theta_PhT) / zh;
    double kpmag = sqrt(pow(kTxdiff, 2) + pow(kTydiff, 2));
    double Ntidle = 0.0;
    double rapidity = log(0.01/xg);
    if (rapidity > 15.80) rapidity = 15.8;
    if (rapidity< 0.0) rapidity = 0.0;
    if (kpmag > maxpT) kpmag = maxpT;
    if (kpmag < minpT) kpmag = minpT;
        if (dipole_model == 0) {
            double G = 0.25 * pow(0.01/xg, 0.29) * exp(0.29 * rapidity); 
            Ntidle = exp(-kpmag*kpmag/G*0.25)/G*0.5; // ????
        } 
        if (dipole_model == 1) {
            int y_index = int(rapidity/Y_step);
            //if ( kpmag < 99.) {
                std::vector<double> xValues3, yValues3;
                xValues3.clear(); yValues3.clear();
                for (int inn= y_index * Klength; inn <  y_index * Klength + Klength; inn++) {
                    xValues3.push_back(pTValues[inn]);
                    yValues3.push_back(F1qgV[inn]);
                
                }
                gsl_interp *interp = gsl_interp_alloc(gsl_interp_cspline, Klength);
                
                gsl_interp_init(interp, xValues3.data(), yValues3.data(), Klength);
                double Ntidle1 = gsl_interp_eval(interp, xValues3.data(), yValues3.data(), kpmag, nullptr);
                /////
                int y_index2 = y_index +1;
                std::vector<double> xValues2, yValues2;
                xValues2.clear(); yValues2.clear();
                for (int inn= y_index2 * Klength; inn <  y_index2 * Klength + Klength; inn++) {
                    xValues2.push_back(pTValues[inn]);
                    yValues2.push_back(F1qgV[inn]);
                
                }
                gsl_interp_init(interp, xValues2.data(), yValues2.data(), Klength);
                double Ntidle2 = gsl_interp_eval(interp, xValues2.data(), yValues2.data(), kpmag, nullptr);
                gsl_interp_free(interp);
                
                Ntidle = (Ntidle2 * (rapidity - YValues[y_index2*Klength-Klength]  ) + 
                      Ntidle1 * (YValues[y_index2*Klength] - rapidity) ) / Y_step;
                Ntidle = Ntidle *2.*M_PI*M_PI/3./kpmag/kpmag;
            //}
        }
        
    if (xg>0.01) {
        double ratio = pow(1.-xg, 4.) / 0.96059601; // 0.96059601 = (1-0.01)^4
        Ntidle =  ratio * Ntidle;
    }
    
    // integrate the whole function
    // Fragmentation functon
    /*
    int is=1, ih=4, ic=0, io=1;
    double u, ub, d, db, s, sb, c, bb, gl;
    fdss_(&is, &ih, &ic, &io, &zh, &Q2, &u, &ub, &d, &db, &s, &sb, &c, &bb, &gl);
    double Wk = 0.0038497433455066264 * (8./9. * u/zh * pdf->xfxQ2(2,xp, Q2) +
                  1./9. * d/zh * pdf->xfxQ2(1,xp, Q2) );
    */
    double Wk = 0.0038497433455066264 * (8./9. * FF->xfxQ2(2, zh, Q2)/zh *  pdf->xfxQ2(2,xp, Q2) +
                  1./9. * FF->xfxQ2(1, zh, Q2)/zh * pdf->xfxQ2(1,xp, Q2) ); 
                  // 0.0038497433455066264 = 3/8/pi**4; 0.02418865082489962 = 3/8/pi**4 * 2 * pi
    double  dsigma_dDeltaphi = 0.025330295910584444 * M_PI * R_Nuclear * R_Nuclear / zh/zh * Wk * pow(kpmag, 2) * Ntidle * sigmahat *
                              kTg * PhT; // Jacobe
                              //0.025330295910584444 = 1/(2pi)^2
    f[0] = dsigma_dDeltaphi * total_volume;
    return 0;
}



int main(int argc, char* argv[]) {
    
    double kgTmin = std::stod(argv[1]);
    double kgTmax = std::stod(argv[2]);
    double phTmin = std::stod(argv[3]);
    double phTmax = std::stod(argv[4]); // Convert the 5th command-line argument to a double
    
    // Set the number of threads to use
    omp_set_num_threads(num_threads);
    
    PARAMETERS params;

    params.etagmax    = 0.35;
    params.etagmin    = -0.35;
    params.kpmagmax    = 1.0;///////////////////
    params.etahmax    = 0.35;
    params.etahmin    = -0.35;
    params.kTgmax    = kgTmax;
    params.kTgmin    = kgTmin;
    params.PhTmax    = phTmax;
    params.PhTmin    = phTmin;
    
    LHAPDF::initPDFSet("CT18NNLO");
    pdf = LHAPDF::mkPDF("CT18NNLO", 0);
    const LHAPDF::PDFSet set("CT18NNLO"); // arxiv: 2101.04664
    // Initialize LHAPDF and set the fragmentation function set
    LHAPDF::initPDFSet("JAM20-SIDIS_FF_hadron_nlo");
    FF = LHAPDF::mkPDF("JAM20-SIDIS_FF_hadron_nlo", 0);  // Use the appropriate member index if there are multiple sets
    

    std::ifstream inputFile("Paul_table/Regularged_FT_proton_MV_Paul");
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
    const int ndim = 6; /* number of dimensions */
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
    //    char output_filename[128];
    std::stringstream filename;
    filename << "dSigma_dDeltaPhi_without_Sub_FBT_MVgamma_"
             << kgTmin << "_" << kgTmax << "_" << phTmin << "_" << phTmax <<".txt";
    ofstream realA(filename.str());
    
    const int length = 160;
    realA << "# Delta_phi   Wk  sigmahat  Ntidle  dSigma_dDeltaPhi";
    realA << endl;
    for (int itheta=0; itheta<length +1; itheta++) {
        params.Delta_phi = itheta * 1. * M_PI / length * 1.; 
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
    
    realA.close();
    return 0;
}

