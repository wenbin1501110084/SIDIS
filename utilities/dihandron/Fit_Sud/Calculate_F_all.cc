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
std::vector<double> YValuesS, rValuesS, DrN, DDrN, DDr_logN_div_logN, DDrNtidle;
const int Klength = 400;
const int Slength = 396;
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
double Fqg1(double bpmag, int iiyy){
    
    std::vector<double> xValues, yValues;
    
    xValues.clear(); yValues.clear();   
      double bbmax = 0.;
    for (int inn= iiyy * Klength; inn < iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    
    bbmax = 0.;
    
    /*
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_cspline, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    */
    /*
    //std::vector<double> YValuesS, rValuesS, DrN, DDrN, DDr_logN_div_logN;
    
    xValues.clear(); yValues.clear();     
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        xValues.push_back(rValuesS[inn]);
        yValues.push_back(DrN[inn]);
     }
    gsl_interp *interp1 = gsl_interp_alloc(gsl_interp_cspline, Slength);
    gsl_interp_init(interp1, xValues.data(), yValues.data(), Slength);
    //double DrN0 = gsl_interp_eval(interp1, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp1);
    */
    yValues.clear();     xValues.clear();
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
    cout << inn << " " << DDrN[inn] << endl;
        yValues.push_back(DDrN[inn]);
        xValues.push_back(rValuesS[inn]);
         if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_cspline, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDrN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    /*
    yValues.clear(); xValues.clear();
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDr_logN_div_logN[inn]);
        xValues.push_back(rValuesS[inn]);
         if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
     }
      if (bpmag > bbmax) {
        return 0.0;
    }
    gsl_interp *interp3 = gsl_interp_alloc(gsl_interp_cspline, Slength);
    gsl_interp_init(interp3, xValues.data(), yValues.data(), Slength);
    double DDr_logN_div_logN0 = gsl_interp_eval(interp3, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp3);
    */
    /*
    yValues.clear();     
    for (int inn=iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDrNtidle[inn]);
     }
    gsl_interp *interp4 = gsl_interp_alloc(gsl_interp_cspline, Slength);
    gsl_interp_init(interp4, xValues.data(), yValues.data(), Slength);
    /double DDrNtidle0 = gsl_interp_eval(interp4, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp4);
    */
    /*
    double alphas_WW = 2./3. * ST / M_PI /M_PI / 2./M_PI  * bpmag *
                        DDr_logN_div_logN0 * (1.-pow(S0, 2.25));
    return alphas_WW;
    */
    double alphas_Fqga = -1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * DDrN0;
    return  alphas_Fqga;
    
    
}


double alphas_Fqga(double bpmag,int iiyy){
    std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();   
    double bbmax = 0.;
    double bbmin = 100.;
    yValues.clear();     xValues.clear();
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDrN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDrN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    
    double alphas_F = -1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * DDrN0;
    return  alphas_F;
}

double alphas_Fqgb(double bpmag,int iiyy){
    std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();   
    double bbmax = 0.;
    double bbmin = 100.;
    for (int inn= iiyy * Klength; inn < iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
        if (rValues[inn] < bbmin) bbmin = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
    yValues.clear();     xValues.clear();
    bbmax = 0.0; bbmin = 1.0;
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDr_logN_div_logN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDr_logN_div_logN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    double alphas_F = 4./6. * ST / M_PI /M_PI * bpmag/ 2./M_PI * 
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) ) * S0;
    return  alphas_F;
}

double alphas_Fgga(double bpmag,int iiyy){
        std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();   
    double bbmax = 10.;
    double bbmin = 1.;
    for (int inn= iiyy * Klength; inn < iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
        if (rValues[inn] < bbmin) bbmin = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
    yValues.clear();     xValues.clear();
    bbmax = 0.0; bbmin = 1.0;
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDrN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDrN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    double alphas_F = -1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * 
                        S0 * DDrN0;
    return  alphas_F;
}

double alphas_Fggb(double bpmag,int iiyy){
    std::vector<double> xValues, yValues;
    yValues.clear();     xValues.clear();
    double bbmax = 0.0; double bbmin = 1.0;
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DrN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DrN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    double alphas_F = 1.5 * ST / M_PI /M_PI * bpmag/ 2./M_PI * 
                        DrN0 * DrN0;
    return  alphas_F;
}

double alphas_Fggc(double bpmag,int iiyy){
        std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();   
    double bbmax = 10.;
    double bbmin = 1.;
    for (int inn= iiyy * Klength; inn < iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
        if (rValues[inn] < bbmin) bbmin = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
    yValues.clear();     xValues.clear();
    bbmax = 0.0; bbmin = 1.0;
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDr_logN_div_logN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
        if (rValues[inn] < bbmin) bbmin = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDr_logN_div_logN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    double alphas_F = 2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI *
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) ) * S0 * S0;
    return  alphas_F;
}

double alphas_Fadj(double bpmag,int iiyy){
    std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();   
    double bbmax = 10.;
    double bbmin = 1.;
    yValues.clear();     xValues.clear();
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDrN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Slength);
    double DDrN0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
    yValues.clear();     xValues.clear();
    bbmax = 10.;
    bbmin = 1.;
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DrN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DrN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    
    
    yValues.clear();     xValues.clear();
    bbmax = 10.;
    bbmin = 1.;
    for (int inn= iiyy * Klength; inn < iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
        if (rValues[inn] < bbmin) bbmin = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp3 = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp3, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp3, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp3);
    
    double DDrNT0 = 2.25 * 1.25 * pow(S0, 0.25) *  DrN0*DrN0 + 2.25 * pow(S0, 1.25) * DDrN0;
    double alphas_F = -2./3. * ST / M_PI /M_PI * bpmag/ 2./M_PI * DDrNT0;
    return  alphas_F;
}

double alphas_FWW(double bpmag,int iiyy){
    std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();   
    double bbmax = 10.;
    double bbmin = 1.;
    for (int inn= iiyy * Klength; inn < iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
        if (rValues[inn] < bbmin) bbmin = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
    yValues.clear();     xValues.clear();
    bbmax = 0.0; bbmin = 1.0;
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDr_logN_div_logN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDr_logN_div_logN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    double alphas_F = 4./6. * ST / M_PI /M_PI * bpmag/ 2./M_PI * 
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) );
    return  alphas_F;
}

double alphas_Fgg6(double bpmag,int iiyy){
    std::vector<double> xValues, yValues;
    xValues.clear(); yValues.clear();   
    double bbmax = 10.;
    double bbmin = 1.;
    for (int inn= iiyy * Klength; inn < iiyy * Klength + Klength; inn++) {
        xValues.push_back(rValues[inn]);
        yValues.push_back(NrValues[inn]);
        if (rValues[inn] > bbmax) bbmax = rValues[inn];
        if (rValues[inn] < bbmin) bbmin = rValues[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, Klength);
    gsl_interp_init(interp, xValues.data(), yValues.data(), Klength);
    double S0 = gsl_interp_eval(interp, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp);
    
    yValues.clear();     xValues.clear();
    bbmax = 0.0; bbmin = 1.0;
    for (int inn= iiyy * Slength; inn <  iiyy * Slength + Slength; inn++) {
        yValues.push_back(DDr_logN_div_logN[inn]);
        xValues.push_back(rValuesS[inn]);
        if (rValuesS[inn] > bbmax) bbmax = rValuesS[inn];
        if (rValuesS[inn] < bbmin) bbmin = rValuesS[inn];
     }
     if (bpmag > bbmax) {
        return 0.0;
    }
    if (bpmag < bbmin) bpmag = bbmin;
    gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_linear, Slength);
    gsl_interp_init(interp2, xValues.data(), yValues.data(), Slength);
    double DDr_logN_div_logN0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmag, nullptr);
    gsl_interp_free(interp2);
    double alphas_F = 4./6. * ST / M_PI /M_PI * bpmag/ 2./M_PI * 
                        DDr_logN_div_logN0 * (1. - pow(S0, 2.25) ) *  pow(S0, 2.25);
    return  alphas_F;
}


    

int main(int argc, char* argv[]) 
{
    // Open the input file
    std::ifstream inputFile("Paul_table/rcBK_MV_Heikki_table.txt");
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
    std::ifstream inputFileS("Do_deriv/Delta_r_N_rcBK_MV_Heikki_table_3.txt");
    if (!inputFileS.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    double Ys, rs, DrNs, DDrNs, DDr_logN_div_logNs, DDrNtidles; // r in [GeV^-1]
    while (inputFileS >> Ys >> rs >> DrNs >> DDrNs >> DDr_logN_div_logNs >> DDrNtidles) {
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
    filename << "F_table_K_p8_MV_all" << startid << "_" << endid <<".txt";
    std::ofstream outputFile(filename.str());
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    
    // Set the number of threads to use
    
    PARAMETERS params;

    
    FBT ogata0 = FBT(0.0, 0, 1000); // Fourier Transform with Jnu, nu=0.0 and N=10
    for (int iy = startid; iy < endid; iy++) {
        for (int ii =0; ii < 4000; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.005;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double Wk = ogata0.fbt(std::bind(alphas_Fqga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fqgb, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fadj, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_FWW, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgg6, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            outputFile << endl;
        }
        for (int ii =0; ii < 2000; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.01 + 20.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double Wk = ogata0.fbt(std::bind(alphas_Fqga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fqgb, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fadj, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_FWW, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgg6, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            outputFile << endl;
        }

        for (int ii =0; ii < 800; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.05 + 40.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double Wk = ogata0.fbt(std::bind(alphas_Fqga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fqgb, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fadj, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_FWW, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgg6, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            outputFile << endl;
        }


        for (int ii =0; ii < 200; ii ++) {
            params.iiyy = iy;
            double qT = ii * 0.1 + 80.;
            params.qqTT = qT;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            double Wk = ogata0.fbt(std::bind(alphas_Fqga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fqgb, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgga, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fadj, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_FWW, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            Wk = ogata0.fbt(std::bind(alphas_Fgg6, std::placeholders::_1, params.iiyy), params.qqTT);
            outputFile << Wk << "  ";
            outputFile << endl;
        }

    }
    outputFile.close();
}

