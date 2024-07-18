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
#include "FBT.h"

std::vector<double> YValues, rValues, NrValues;
std::vector<double> YValuesS, rValuesS, DrN, DDrN, DDr_logN_div_logN, DDrNtidle, K2;
const int Klength = 401;
const int Slength = 397;
const double ST = 1.0;//M_PI * 6.6*6.6 /0.197/0.197;// GeV^-2
const double betaSud = 0.7192823;
const double Lambda2Sud = 0.03745153;
const long long int sample_points = 50000000;
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
    double bpmag = x[0] * helper->bmax + 1.09673e-06;
    std::vector<double> xValues,yValues, yValues2;
    xValues.clear(); yValues.clear();   
    for (int inn= helper->iiyy * Klength; inn <  helper->iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        
     }
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_akima, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
    
    xValues.clear(); yValues.clear(); yValues2.clear();       
    for (int inn= helper->iiyy * Slength; inn <  helper->iiyy * Slength + Slength; inn++) {
        xValues.push_back(rValuesS[inn]);
        yValues.push_back(K2[inn]);
        yValues2.push_back(DDr_logN_div_logN[inn]);
     }
    gsl_interp *interp1 = gsl_interp_alloc(gsl_interp_akima, Slength);
    gsl_interp_init(interp1, xValues.data(), yValues.data(), Slength);
    double K20 = gsl_interp_eval(interp1, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp1);
    
    double alphas_HWW2 = 4./6. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(2, bpmag *helper->qqTT ) *
                        K20 * (1. - pow(S0, 2.25) );
    f[0] = alphas_HWW2 * helper->bmax;
    
    
    gsl_interp *interp3 = gsl_interp_alloc(gsl_interp_akima, Slength);
    gsl_interp_init(interp3, xValues.data(), yValues2.data(), Slength);
    double DDr_logN_div_logN0 = gsl_interp_eval(interp3, xValues.data(), yValues2.data(), bpmag, nullptr);
    gsl_interp_free(interp3);
    double alphas_WWd = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1.-pow(S0, 2.25));
    f[1] = alphas_WWd  * helper->bmax;;
    
    /*
    
    xValues.clear(); yValues.clear();     
    for (int inn= helper->iiyy * Slength; inn <  helper->iiyy * Slength + Slength; inn++) {
        xValues.push_back(rValuesS[inn]);
        yValues.push_back(DrN[inn]);
     }
    gsl_interp *interp1 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp1, xValues.data(), yValues.data(), Slength);
    double DrN0 = gsl_interp_eval(interp1, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp1);
    
    yValues.clear();     
    for (int inn= helper->iiyy * Slength; inn <  helper->iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDrN[inn]);
     }
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDrN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    
    yValues.clear();     
    for (int inn= helper->iiyy * Slength; inn <  helper->iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDr_logN_div_logN[inn]);
     }
    gsl_interp *interp3 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp3, xValues.data(), yValues.data(), Slength);
    double DDr_logN_div_logN0 = gsl_interp_eval(interp3, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp3);
    */
    /*
    yValues.clear();     
    for (int inn= helper->iiyy * Slength; inn <  helper->iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDrNtidle[inn]);
     }
    gsl_interp *interp4 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp4, xValues.data(), yValues.data(), Slength);
    double DDrNtidle0 = gsl_interp_eval(interp4, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp4);
    */
    /*
    double alphas_Fqga = -1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDrN0;
    f[0] = alphas_Fqga * helper->bmax;
    
    double alphas_Fqgb = 4./6. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) ) * S0;
    f[1] = alphas_Fqgb * helper->bmax;
    
    double alphas_Fgga = -1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        S0 * DDrN0;
    f[2] = alphas_Fgga * helper->bmax;
    
    double alphas_Fggb = 1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DrN0 * DrN0;
    f[3] = alphas_Fggb * helper->bmax;
    
    double alphas_Fggc = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) ) * S0 * S0;
    f[4] = alphas_Fggc * helper->bmax;
    
    double alphas_Fadj = -2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * 2.25 * 1.25 * pow(S0, 0.25) *  DrN0*DrN0 + 2.25 * pow(S0, 1.25) * DDrN0 *  boost::math::cyl_bessel_j(0, bpmag *helper->qqTT );
    f[5] = alphas_Fadj * helper->bmax;
    
    double alphas_WW = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1.-pow(S0, 2.25));
    f[6] = alphas_WW * helper->bmax;
    
    
    double alphas_gg6 = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1.-pow(S0, 2.25)) * pow(S0, 2.25);
    f[7] = alphas_gg6 * helper->bmax;
    */
    return 0;
}

    

int main(int argc, char* argv[]) 
{
    // Open the input file
    std::ifstream inputFile("Paul_table/rcBK-eta-etaf_16-rc_Balitsky-C_3.8-init_MV_Qs00.789937.res");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    int startid = std::stoi(argv[1]);
    int endid = std::stoi(argv[2]);
    // Read data from the file
    double Y, r, Nr; // r in [GeV^-1]
    while (inputFile >> Y >> r >> Nr) {
        YValues.push_back(Y);
        rValues.push_back(r);
        NrValues.push_back( 1.-Nr);
    }
    // Close the file
    inputFile.close();
    
    // Open the input file
    std::ifstream inputFileS("Paul_table/Delta_r_N_rcBK-eta-etaf_16-rc_Balitsky-C_3.8-init_MV_Qs00.789937-9.txt");
    if (!inputFileS.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    double Ys, rs, DrNs, DDrNs, DDr_logN_div_logNs, DDrNtidles, K2s; // r in [GeV^-1]
    while (inputFileS >> Ys >> rs >> DrNs >> DDrNs >> DDr_logN_div_logNs >> DDrNtidles >> K2s) {
        YValuesS.push_back(Ys);
        rValuesS.push_back(rs);
        DrN.push_back(DrNs);
        DDrN.push_back(DDrNs);
        DDr_logN_div_logN.push_back(DDr_logN_div_logNs);
        DDrNtidle.push_back(DDrNtidles);
        K2.push_back(K2s);
    }
    // Close the file
    inputFileS.close();
    
    
    // Open the output file
    //std::ofstream outputFile("rcNK_table_K.txt");
        std::stringstream filename;
    filename << "FWW2_table_K_p8_MV_" << startid << "_" << endid <<".txt";
    std::ofstream outputFile(filename.str());
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    
    // Set the number of threads to use
    
    PARAMETERS params;

    params.bmax      = 60.;// GeV^-1
    const int num_threads = 10;
    // Set the number of threads to use
    omp_set_num_threads(num_threads);
    /* Define the integration parameters */
    const int ndim = 1; /* number of dimensions */
    const int ncomp = 2; /* number of components */
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
    
    
    for (int iy = startid; iy < endid; iy++) {
        for (int ii =0; ii < 400; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.05;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            outputFile << integral[0]  << "  " << integral[1] << endl;
        }
        for (int ii =0; ii < 200; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.1 + 20.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart,
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            outputFile << integral[0]  << "  " << integral[1] << endl;
        }

        for (int ii =0; ii < 80; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.5 + 40.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart,
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            outputFile << integral[0]  << "  " << integral[1] << endl;
        }


        for (int ii =0; ii < 20; ii ++) {
            params.iiyy = iy;
            double qT = ii * 1. + 80.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart,
                        nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
           outputFile << integral[0]  << "  " << integral[1] << endl;
        }

    }
    outputFile.close();
}

