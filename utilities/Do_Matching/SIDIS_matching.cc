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
const long long int sample_points = 200000;
const long long int sample_points0 = 10000;
const double R_Nuclear = 6.2;//fm
const int num_threads = 6;
const double rootsnn = 200.;// GeV
const double Rcut = 0.3;
const int dipole_model = 1; //1: rcBK; 0: GBW
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



int matching(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    // x0: kT2;
    double kT2 = x[0] * helper->Q2;
    double alpha_s = pdf->alphasQ2( kT2);// call from LHAPDF
    double C =  helper->Rp2 *3. / 16. / M_PI / M_PI / alpha_s;// Rp in GeV^-1
    // N tidle // GBW, photon-nucleon
    double Ntidle;
    if (dipole_model == 0 ) {
        //double G = 0.25 * pow(0.0003/xg, 0.29);
        double xg = 0.01; // Matching point 
        double rapidity = log(helper->x0/xg);
        double G = 0.25 * pow(helper->x0/xg, 0.29) * exp(0.29 * rapidity); 
        Ntidle = exp(-kT2/G*0.25)/G*0.5; // ????
    } else {
        if (dipole_model == 1 ) {
            double Am = 3.995189967514155782e+00;
            double Bm =  2.031462982535007011e+00/2.;
            double Cm =  4.273606519780273949e+00;
            double Dm =  4.415365535915241502e+00;
            double Em = 1.083775271617884867e-01;
            double Fm =  2.313193722087994786e+00;
            
            Ntidle = Am * exp(-1. * pow(kT2, Bm) * Cm + Dm) + Em * pow(kT2, Fm); //A * exp(-k**B * C + D) + E*k**F
        }
    }
    
    f[0] = C * Ntidle * kT2 * helper->Q2;
    return 0;
}



int main(int argc, char* argv[]) {

    // Set the number of threads to use
    omp_set_num_threads(num_threads);
    
    PARAMETERS params;

    
    /* Define the integration parameters */
    const int ndim = 1; /* number of dimensions */
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
    
    // Initialize LHAPDF
    LHAPDF::initPDFSet("cteq66");
    // Access PDF set
    pdf = LHAPDF::mkPDF("cteq66", 0);
    const LHAPDF::PDFSet set("cteq66"); // arxiv: 2101.04664

    // Initialize LHAPDF and set the fragmentation function set
    LHAPDF::initPDFSet("JAM20-SIDIS_FF_hadron_nlo");
    FF = LHAPDF::mkPDF("JAM20-SIDIS_FF_hadron_nlo", 0);  // Use the appropriate member index if there are multiple sets

    params.Rp2 = 1.; 
    //output the results to file
    char output_filename[128];
    sprintf(output_filename,"Matching_output");
    ofstream realA(output_filename);
    
    const int length = 2000;
    realA << "# Q^2   xfxQ2   C_integrate_at_x0";
    realA << endl;
    for (int itheta=100; itheta<length; itheta++) {
        params.Q2 = itheta * 0.01; 
        realA << params.Q2 << "  ";
        /* Call the integrator */
        params.x0 = 0.0098;
        realA << pdf->xfxQ2(21,params.x0, params.Q2) << "  "; 
        llVegas(ndim, ncomp, matching, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
        realA << integral[0] << "  ";
        
        params.x0 = 0.01;
        realA << pdf->xfxQ2(21,params.x0, params.Q2) << "  "; 
        llVegas(ndim, ncomp, matching, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
        realA << integral[0] << "  ";
        
        params.x0 = 0.0102;
        realA << pdf->xfxQ2(21,params.x0, params.Q2) << "  "; 
        llVegas(ndim, ncomp, matching, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
        realA << integral[0] << "  ";
        realA << endl;
        
    }
    return 0;
}

