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
#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>

#include <complex>
#include <vector>
#include "FBT.h"

std::vector<double> YValues, rValues, NrValues;
std::vector<double> YValuesS, rValuesS, DrN, DDrN, DDr_logN_div_logN, DDrNtidle;
const int Klength = 401;
const int Slength = 397;
const double ST = 1.0;//M_PI * 6.6*6.6 /0.197/0.197;// GeV^-2
const double betaSud = 0.7192823;
const double Lambda2Sud = 0.03745153;
const long long int sample_points = 500000;
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
    double bmin;
    double j0zeros0;
    double j0zeros1;
    int iiyy;
};


using namespace std;

struct BesselParams {
    double (*function)(double, void *);
    void *params;
};

double bessel_j0(double x, void *params) {
    return gsl_sf_bessel_J0(x);
}

double find_zero(double x_lower, double x_upper) {
    const gsl_root_fsolver_type *T;
    gsl_root_fsolver *s;
    gsl_function F;
    struct BesselParams p = {&bessel_j0, nullptr};

    F.function = p.function;
    F.params = p.params;

    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc(T);
    gsl_root_fsolver_set(s, &F, x_lower, x_upper);

    int status;
    double r;
    int iter = 0, max_iter = 100;

    do {
        iter++;
        status = gsl_root_fsolver_iterate(s);
        r = gsl_root_fsolver_root(s);
        x_lower = gsl_root_fsolver_x_lower(s);
        x_upper = gsl_root_fsolver_x_upper(s);
        status = gsl_root_test_interval(x_lower, x_upper, 0, 1e-7);
    } while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fsolver_free(s);

    if (status != GSL_SUCCESS) {
        std::cerr << "Failed to find root" << std::endl;
        return NAN;
    }

    return r;
}


int zh_kp2_b_integrated(const int *ndim, const cubareal *x, const int *ncomp, cubareal *f, void *userdata){
    PARAMETERS *helper = (PARAMETERS*)userdata;
    //double bpmag = x[0] * helper->bmax + 1.09673e-06;
    double bpmag = x[0] * (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT) + helper->j0zeros0/helper->qqTT;
    
    double bminn1 = 1.; double bmaxx1 = 1.;
    for (int inn= helper->iiyy * Klength; inn <  helper->iiyy * Klength + Klength; inn++) {
        if (rValues[inn] < bminn1) bminn1 = rValues[inn];
        if (rValues[inn] > bmaxx1) bmaxx1 = rValues[inn];
     }
     
     double bminn2 = 1.; double bmaxx2 = 1.;
    for (int inn= helper->iiyy * Slength; inn <  helper->iiyy * Slength + Slength; inn++) {
        if (rValuesS[inn] < bminn2) bminn2 = rValuesS[inn];
        if (rValuesS[inn] > bmaxx2) bmaxx2 = rValuesS[inn];
     }
     if (bpmag < std::max(bminn1, bminn2)) bpmag = std::max(bminn1, bminn2);
     if (bpmag > std::min(bmaxx1, bmaxx2)) bpmag = std::min(bmaxx1, bmaxx2);
     
    std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();     
    for (int inn= helper->iiyy * Klength; inn <  helper->iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
     }
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
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
    double alphas_Fqga = -1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDrN0;
    f[0] = alphas_Fqga *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    double alphas_Fqgb = 4./6. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) ) * S0;
    f[1] = alphas_Fqgb *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    double alphas_Fgga = -1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        S0 * DDrN0;
    f[2] = alphas_Fgga *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    double alphas_Fggb = 1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DrN0 * DrN0;
    f[3] = alphas_Fggb *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    double alphas_Fggc = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) ) * S0 * S0;
    f[4] = alphas_Fggc *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    double alphas_Fadj = -2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * 2.25 * 1.25 * pow(S0, 0.25) *  DrN0*DrN0 + 2.25 * pow(S0, 1.25) * DDrN0 *  boost::math::cyl_bessel_j(0, bpmag *helper->qqTT );
    f[5] = alphas_Fadj *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    double alphas_WW = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1.-pow(S0, 2.25));
    f[6] = alphas_WW *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    
    double alphas_gg6 = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * boost::math::cyl_bessel_j(0, bpmag *helper->qqTT ) *
                        DDr_logN_div_logN0 * (1.-pow(S0, 2.25)) * pow(S0, 2.25);
    f[7] = alphas_gg6 *  (helper->j0zeros1/helper->qqTT - helper->j0zeros0/helper->qqTT);
    
    return 0;
}

    

int main(int argc, char* argv[]) 
{
    // Open the input file
    std::ifstream inputFile("Paul_table/rcBK-eta-etaf_16-rc_Balitsky-C_3.8-init_MV_Qs00.32249.res");
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
    std::ifstream inputFileS("Paul_table/Delta_r_N_rcBK-eta-etaf_16-rc_Balitsky-C_3.8-init_MV_Qs00.32249_8.txt");
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
    }
    // Close the file
    inputFileS.close();
    
    
    // Open the output file
    //std::ofstream outputFile("rcNK_table_K.txt");
        std::stringstream filename;
    filename << "F_table_K_p8_MVe_" << startid << "_" << endid <<".txt";
    std::ofstream outputFile(filename.str());
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    
    // Set the number of threads to use
    
    PARAMETERS params;

    params.bmax      = 91.;// GeV^-1
    params.bmin      = 1.09e-06;
    const int num_threads = 10;
    // Set the number of threads to use
    omp_set_num_threads(num_threads);
    /* Define the integration parameters */
    const int ndim = 1; /* number of dimensions */
    const int ncomp = 8; /* number of components */
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
    
    const int num_zeros = 50;
    double Jozeros_array[num_zeros+1] = {0.};
    double initial_guess_lower = 2.0;
    double initial_guess_upper = 5.0;
    for (int i = 0; i < num_zeros; ++i) {
        double zero_temp = find_zero(initial_guess_lower, initial_guess_upper);
        initial_guess_lower = zero_temp + 1.0;
        initial_guess_upper = initial_guess_lower + 3.0;
        Jozeros_array[i+1] = zero_temp;
    }
    double ncuts = 1.e-5;
    for (int iy = startid; iy < endid; iy++) {
        for (int ii =0; ii < 400; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.05;
            if (ii == 0) qT = 1.e-5;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double FTMDs[8] = {0., 0., 0., 0., 0., 0.,0.,0.};
            params.j0zeros0 = 0.;
            params.j0zeros1 = Jozeros_array[1];
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                     nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            for (int ikk=0; ikk<8; ikk++) {
                FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
            }
            for (int i = 1; i < num_zeros; ++i) {
                params.j0zeros0 = Jozeros_array[i];
                if (params.j0zeros0/params.qqTT > params.bmax) {
                    break;
                }
                
                
                params.j0zeros1 = Jozeros_array[i+1];
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                for (int ikk=0; ikk<8; ikk++) {
                    FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
                }
                bool condition_met = true;

                for (int i = 0; i < num_zeros; ++i) {
                    if (std::abs(integral[i] / FTMDs[i]) >= ncuts) {
                        condition_met = false;
                        break;
                    }
                }

                if (condition_met) break;

                     
            }
            outputFile << FTMDs[0] << "  " << FTMDs[1] << "  " << FTMDs[2] << "  " << FTMDs[3] << "  " << FTMDs[4]
                       << "  "  << FTMDs[5] << "  " << FTMDs[6] << "  " << FTMDs[7] << endl;
        }
        
        for (int ii =0; ii < 200; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.1 + 20.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double FTMDs[8] = {0., 0., 0., 0., 0., 0.,0.,0.};
            params.j0zeros0 = 0.;
            params.j0zeros1 = Jozeros_array[1];
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                     nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            for (int ikk=0; ikk<8; ikk++) {
                FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
            }
            for (int i = 1; i < num_zeros; ++i) {
                params.j0zeros0 = Jozeros_array[i];
                if (params.j0zeros0/params.qqTT > params.bmax) {
                    break;
                }
                
                params.j0zeros1 = Jozeros_array[i+1];
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                for (int ikk=0; ikk<8; ikk++) {
                    FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
                }
                
                bool condition_met = true;

                for (int i = 0; i < num_zeros; ++i) {
                    if (std::abs(integral[i] / FTMDs[i]) >= ncuts) {
                        condition_met = false;
                        break;
                    }
                }

                if (condition_met) break;
            }
            outputFile << FTMDs[0] << "  " << FTMDs[1] << "  " << FTMDs[2] << "  " << FTMDs[3] << "  " << FTMDs[4]
                       << "  "  << FTMDs[5] << "  " << FTMDs[6] << "  " << FTMDs[7] << endl;
                       
        }

        for (int ii =0; ii < 80; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.5 + 40.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double FTMDs[8] = {0., 0., 0., 0., 0., 0.,0.,0.};
            params.j0zeros0 = 0.;
            params.j0zeros1 = Jozeros_array[1];
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                     nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            for (int ikk=0; ikk<8; ikk++) {
                FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
            }
            for (int i = 1; i < num_zeros; ++i) {
                params.j0zeros0 = Jozeros_array[i];
                if (params.j0zeros0/params.qqTT > params.bmax) {
                    break;
                }
                
                
                params.j0zeros1 = Jozeros_array[i+1];
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                for (int ikk=0; ikk<8; ikk++) {
                    FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
                }
                
                bool condition_met = true;

                for (int i = 0; i < num_zeros; ++i) {
                    if (std::abs(integral[i] / FTMDs[i]) >= ncuts) {
                        condition_met = false;
                        break;
                    }
                }

                if (condition_met) break;
                     
            }
            outputFile << FTMDs[0] << "  " << FTMDs[1] << "  " << FTMDs[2] << "  " << FTMDs[3] << "  " << FTMDs[4]
                       << "  "  << FTMDs[5] << "  " << FTMDs[6] << "  " << FTMDs[7] << endl;
                       
        }


        for (int ii =0; ii < 20; ii ++) {
            params.iiyy = iy;
            double qT = ii * 1. + 80.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double FTMDs[8] = {0., 0., 0., 0., 0., 0.,0.,0.};
            params.j0zeros0 = 0.;
            params.j0zeros1 = Jozeros_array[1];
            llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                     nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
            for (int ikk=0; ikk<8; ikk++) {
                FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
            }
            for (int i = 1; i < num_zeros; ++i) {
                params.j0zeros0 = Jozeros_array[i];
                if (params.j0zeros0/params.qqTT > params.bmax) {
                    break;
                }
                
                params.j0zeros1 = Jozeros_array[i+1];
                llVegas(ndim, ncomp, zh_kp2_b_integrated, &params, nvec, epsrel, epsabs, flags, seed, mineval, maxeval, nstart, 
                            nincrease, nbatch, gridno, NULL, NULL, &neval, &fail, integral, error, prob);
                for (int ikk=0; ikk<8; ikk++) {
                    FTMDs[ikk] = FTMDs[ikk] + integral[ikk];
                }
                bool condition_met = true;

                for (int i = 0; i < num_zeros; ++i) {
                    if (std::abs(integral[i] / FTMDs[i]) >= ncuts) {
                        condition_met = false;
                        break;
                    }
                }

                if (condition_met) break;
            }
            outputFile << FTMDs[0] << "  " << FTMDs[1] << "  " << FTMDs[2] << "  " << FTMDs[3] << "  " << FTMDs[4]
                       << "  "  << FTMDs[5] << "  " << FTMDs[6] << "  " << FTMDs[7] << endl;
                       
        }
        

    }
    outputFile.close();
}

