#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <cmath>
#include <math.h>
#include <complex>
#include <vector>
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

// g++ Do_Deriv_r.cc -lcuba -lgsl -lgslcblas -lm -fopenmp -o Diffcuba

std::vector<double> YValues, rValues, NrValues;
const int Klength = 400;
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

    

int main(int argc, char* argv[]) 
{
    // Open the input file
    std::ifstream inputFile("../rcNK_table/rcBK_MV_Heikki_table.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }
    double Y, r, Nr; // r in [GeV^-1]
    while (inputFile >> Y >> r >> Nr) {
        YValues.push_back(Y);
        rValues.push_back(r);
        NrValues.push_back( 1.-Nr);
    }
    // Close the file
    inputFile.close();
    
    // Open the output file
    std::stringstream filename;
    filename << "Delta_r_N_rcBK_MV_Heikki_table_3.txt";
    std::ofstream outputFile(filename.str());
    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    
    const double dy = 0.2;
    int leny = 16/dy;
    for (int iy = 0; iy < leny; iy++) {
        std::vector<double> xValues, yValues, yValues2p25;
        xValues.clear(); yValues.clear();   yValues2p25.clear();  
        for (int inn= iy * Klength; inn <  iy * Klength + Klength; inn++) {
            xValues.push_back(rValues[inn]);
            yValues.push_back(NrValues[inn]);
            yValues2p25.push_back(pow(NrValues[inn], 2.25));
        }
        gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_cspline, Klength);
        gsl_interp_accel *acc = gsl_interp_accel_alloc(); 
        gsl_interp_init(interp2, xValues.data(), yValues.data(), Klength);
        double bpmagtemp;
        for (int inn= iy * Klength+2; inn <  iy * Klength + Klength-2; inn++) {
            double dh = (rValues[inn+1] - rValues[inn])/3.;
            bpmagtemp = rValues[inn] + dh * 2.;
            double NrValuesp2 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] + dh * 1.;
            double NrValuesp1 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 1.;
            double NrValuesm1 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 2.;
            double NrValuesm2 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            double DrN = (NrValuesm2 - 8. * NrValuesm1 + 8. * NrValuesp1 - NrValuesp2) / 12./dh;
            
            bpmagtemp = rValues[inn] + dh * 4.;
            double NrValuesp4 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] + dh * 3.;
            double NrValuesp3 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn];
            double NrValuesp0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 3.;
            double NrValuesm3 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 4.;
            double NrValuesm4 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), bpmagtemp, acc);
            
            double DrNm2 = (NrValuesm4 - 8. * NrValuesm3 + 8. * NrValuesm1 - NrValuesp0) / 12./dh;
            double DrNm1 = (NrValuesm3 - 8. * NrValuesm2 + 8. * NrValuesp0 - NrValuesp1) / 12./dh;
            double DrNp1 = (NrValuesm1 - 8. * NrValuesp0 + 8. * NrValuesp2 - NrValuesp3) / 12./dh;
            double DrNp2 = (NrValuesp0 - 8. * NrValuesp1 + 8. * NrValuesp3 - NrValuesp4) / 12./dh;
            
            double ppdr_temp = (DrNm2 - 8. * DrNm1 + 8. * DrNp1 - DrNp2) / 12./dh;
            double DDrN = DrN / rValues[inn] + ppdr_temp;
            double DDr_logN_div_logN = (DDrN * NrValues[inn] - DrN*DrN ) / (NrValues[inn]* NrValues[inn] * log(NrValues[inn]));
            /*
            double NrValuesp21 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagp1, acc);
            double DrNtildep1 = ( NrValuesp21 - pow(NrValues[inn], 2.25) ) / (bpmagp1 - rValues[inn]);
            double NrValuesm21 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagm1, acc);
            double DrNtildem1 = ( NrValuesm21 - pow(NrValues[inn], 2.25)) / (bpmagm1 - rValues[inn]);
            double DrNtilde = DrNtildep1;
            double DDrNtidle = DrNtilde / rValues[inn] + (DrNtildep1 - DrNtildem1) / (rValues[inn] - bpmagm1);
            */
            bpmagtemp = rValues[inn] + dh * 2.;
            double TNrValuesp2 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] + dh * 1.;
            double TNrValuesp1 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 1.;
            double TNrValuesm1 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 2.;
            double TNrValuesm2 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            double TDrN = (TNrValuesm2 - 8. * TNrValuesm1 + 8. * TNrValuesp1 - TNrValuesp2) / 12./dh;
            
            bpmagtemp = rValues[inn] + dh * 4.;
            double TNrValuesp4 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] + dh * 3.;
            double TNrValuesp3 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn];
            double TNrValuesp0 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 3.;
            double TNrValuesm3 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            bpmagtemp = rValues[inn] - dh * 4.;
            double TNrValuesm4 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagtemp, acc);
            
            double TDrNm2 = (TNrValuesm4 - 8. * TNrValuesm3 + 8. * TNrValuesm1 - TNrValuesp0) / 12./dh;
            double TDrNm1 = (TNrValuesm3 - 8. * TNrValuesm2 + 8. * TNrValuesp0 - TNrValuesp1) / 12./dh;
            double TDrNp1 = (TNrValuesm1 - 8. * TNrValuesp0 + 8. * TNrValuesp2 - TNrValuesp3) / 12./dh;
            double TDrNp2 = (TNrValuesp0 - 8. * TNrValuesp1 + 8. * TNrValuesp3 - TNrValuesp4) / 12./dh;
            
            double Tppdr_temp = (TDrNm2 - 8. * TDrNm1 + 8. * TDrNp1 - TDrNp2) / 12./dh;
            double DDrNtidle = TDrN / rValues[inn] + Tppdr_temp;
            
            if (NrValues[inn] == 0) DDr_logN_div_logN  = 0.0;
            outputFile << iy * 0.2 << "  " << rValues[inn] << "  " << DrN << "  " << DDrN << "  " << DDr_logN_div_logN << "  " << DDrNtidle 
                       << endl;
        }
        gsl_interp_free(interp2);
        gsl_interp_accel_free(acc);
    }
    outputFile.close();
}

