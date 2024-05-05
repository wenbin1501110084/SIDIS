// bessel.cpp
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


std::vector<double> YValues, rValues, NrValues;
const int Klength = 400;
const long long int sample_points = 10000000;
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
};


using namespace std;
int zh_kp2_b_integrated(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    // x0: zh; x1: kp; x2: thetakp; x3 etagamma; x4: etah; x5: kTgamma; x6: PhT; x7: theta_kT
    double bpmag = x[0] * helper->bmax;
    double xValues[Klength], yValues[Klength];
    int ic = 0;
    for (int inn= helper->iiyy * Klength; inn <  helper->iiyy * Klength + Klength; inn++) {
        xValues[ic] =(rValues[inn]);
        yValues[ic] =(NrValues[inn]);
        ic++;
     }
                    
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp, xValues, yValues, Klength);
    double Ntidle0 = gsl_interp_eval(interp, xValues, yValues, bpmag, nullptr);
    gsl_interp_free(interp);
    double Wk = 2. * M_PI * bpmag * Ntidle0 *  boost::math::cyl_bessel_j(0, bpmag *helper->qqTT );
    f[0] = Wk * helper->bmax;
    return 0;
}

    

int main( void )
{
    // Open the input file
    std::ifstream inputFile("rcBK_MV_Heikki_table.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Read data from the file
    std::vector<double> Integrated_NrValues;
    double Y, r, Nr; // r in [GeV^-1]
    while (inputFile >> Y >> r >> Nr) {
        YValues.push_back(Y);
        rValues.push_back(r);
        NrValues.push_back(1. - Nr);
    }
    // Close the file
    inputFile.close();
    
    // read in integrated Nr
    std::ifstream inputFileNr("integrated_Nr_MV");
    if (!inputFileNr.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    double int_Nr;
    while (inputFileNr >> int_Nr) {
        Integrated_NrValues.push_back(int_Nr);
    }
    inputFileNr.close();
    
    // Open the output file
    std::ofstream outputFile("rcNK_table_K.txt");
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    
    // Set the number of threads to use
    
    PARAMETERS params;

    params.bmax      = 10.;// GeV^-1
    const int num_threads = 6;
    // Set the number of threads to use
    omp_set_num_threads(num_threads);
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
    
    
    for (int iy = 0; iy < 81; iy++) {
        for (int ii =0; ii < 1000; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.01;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            // Create a subset vector using iterators
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            //if (ii ==0) {
            //    res = Integrated_NrValues[iy];
           // }
            outputFile << integral[0] << endl;
        }
    }
    outputFile.close();
}
