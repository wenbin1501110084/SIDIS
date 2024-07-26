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
const int Klength = 401;
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
    std::ifstream inputFile("../Paul_table/rcBK-eta-etaf_16-rc_Balitsky-C_3.8-init_MV_Qs00.32249.res");
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
    filename << "Delta_r_N_rcBK-eta-etaf_16-rc_Balitsky-C_3.8-init_MV_Qs00.32249.txt";
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
            xValues.push_back(log(rValues[inn]));
            yValues.push_back(log(abs(NrValues[inn])));
            yValues2p25.push_back(log(abs(pow(NrValues[inn], 2.25))));
        }
        gsl_interp *interp2 = gsl_interp_alloc(gsl_interp_akima, Klength); //gsl_interp_akima_periodic, gsl_interp_akima
        gsl_interp_accel *acc = gsl_interp_accel_alloc(); 
        gsl_interp_init(interp2, xValues.data(), yValues.data(), Klength);
        double bpmagtemp;
        for (int inn= iy * Klength+2; inn <  iy * Klength + Klength-2; inn++) {
            double dh = (rValues[inn+1] - rValues[inn])/3.;
            bpmagtemp = rValues[inn] + dh * 2.;
            double NrValuesp2 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesp2 = exp(NrValuesp2);
            bpmagtemp = rValues[inn] + dh * 1.;
            double NrValuesp1 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesp1 = exp(NrValuesp1);
            bpmagtemp = rValues[inn] - dh * 1.;
            double NrValuesm1 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesm1 = exp(NrValuesm1);
            bpmagtemp = rValues[inn] - dh * 2.;
            double NrValuesm2 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesm2 = exp(NrValuesm2);
            double DrN = (NrValuesm2 - 8. * NrValuesm1 + 8. * NrValuesp1 - NrValuesp2) / 12./dh;
            
            bpmagtemp = rValues[inn] + dh * 4.;
            double NrValuesp4 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesp4 = exp(NrValuesp4);
            bpmagtemp = rValues[inn] + dh * 3.;
            double NrValuesp3 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesp3 = exp(NrValuesp3);
            bpmagtemp = rValues[inn];
            double NrValuesp0 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesp0 = exp(NrValuesp0);
            bpmagtemp = rValues[inn] - dh * 3.;
            double NrValuesm3 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesm3 = exp(NrValuesm3);
            bpmagtemp = rValues[inn] - dh * 4.;
            double NrValuesm4 = gsl_interp_eval(interp2, xValues.data(), yValues.data(), log(bpmagtemp), acc);
            NrValuesm4 = exp(NrValuesm4);
            
            double DrNm2 = (NrValuesm4 - 8. * NrValuesm3 + 8. * NrValuesm1 - NrValuesp0) / 12./dh;
            double DrNm1 = (NrValuesm3 - 8. * NrValuesm2 + 8. * NrValuesp0 - NrValuesp1) / 12./dh;
            double DrNp1 = (NrValuesm1 - 8. * NrValuesp0 + 8. * NrValuesp2 - NrValuesp3) / 12./dh;
            double DrNp2 = (NrValuesp0 - 8. * NrValuesp1 + 8. * NrValuesp3 - NrValuesp4) / 12./dh;
            
            double ppdr_temp = (DrNm2 - 8. * DrNm1 + 8. * DrNp1 - DrNp2) / 12./dh;
            double DDrN = DrN / rValues[inn] + ppdr_temp;
            double DDr_m_Dr_dr_N_div_logN = ( ppdr_temp / NrValuesp0 - DrN * DrN / NrValuesp0 / NrValuesp0 -
                                              DrN / NrValuesp0 / rValues[inn] )/ log(NrValuesp0);
            double DDr_logN_div_logN = (DDrN * NrValuesp0 - DrN*DrN ) / (NrValuesp0* NrValuesp0 * log(NrValuesp0));
            /*
            double NrValuesp21 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagp1, acc);
            double DrNtildep1 = ( NrValuesp21 - pow(NrValues[inn], 2.25) ) / (bpmagp1 - rValues[inn]);
            double NrValuesm21 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), bpmagm1, acc);
            double DrNtildem1 = ( NrValuesm21 - pow(NrValues[inn], 2.25)) / (bpmagm1 - rValues[inn]);
            double DrNtilde = DrNtildep1;
            double DDrNtidle = DrNtilde / rValues[inn] + (DrNtildep1 - DrNtildem1) / (rValues[inn] - bpmagm1);
            */
            bpmagtemp = rValues[inn] + dh * 2.;
            double TNrValuesp2 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesp2 = exp(TNrValuesp2);
            bpmagtemp = rValues[inn] + dh * 1.;
            double TNrValuesp1 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesp1 = exp(TNrValuesp1);
            bpmagtemp = rValues[inn] - dh * 1.;
            double TNrValuesm1 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesm1 = exp(TNrValuesm1);
            bpmagtemp = rValues[inn] - dh * 2.;
            double TNrValuesm2 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesm2 = exp(TNrValuesm2);
            double TDrN = (TNrValuesm2 - 8. * TNrValuesm1 + 8. * TNrValuesp1 - TNrValuesp2) / 12./dh;
            
            bpmagtemp = rValues[inn] + dh * 4.;
            double TNrValuesp4 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesp4 = exp(TNrValuesp4);
            bpmagtemp = rValues[inn] + dh * 3.;
            double TNrValuesp3 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesp3 = exp(TNrValuesp3);
            bpmagtemp = rValues[inn];
            double TNrValuesp0 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesp0 = exp(TNrValuesp0);
            bpmagtemp = rValues[inn] - dh * 3.;
            double TNrValuesm3 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesm3 = exp(TNrValuesm3);
            bpmagtemp = rValues[inn] - dh * 4.;
            double TNrValuesm4 = gsl_interp_eval(interp2, xValues.data(), yValues2p25.data(), log(bpmagtemp), acc);
            TNrValuesm4 = exp(TNrValuesm4);
            
            double TDrNm2 = (TNrValuesm4 - 8. * TNrValuesm3 + 8. * TNrValuesm1 - TNrValuesp0) / 12./dh;
            double TDrNm1 = (TNrValuesm3 - 8. * TNrValuesm2 + 8. * TNrValuesp0 - TNrValuesp1) / 12./dh;
            double TDrNp1 = (TNrValuesm1 - 8. * TNrValuesp0 + 8. * TNrValuesp2 - TNrValuesp3) / 12./dh;
            double TDrNp2 = (TNrValuesp0 - 8. * TNrValuesp1 + 8. * TNrValuesp3 - TNrValuesp4) / 12./dh;
            
            double Tppdr_temp = (TDrNm2 - 8. * TDrNm1 + 8. * TDrNp1 - TDrNp2) / 12./dh;
            double DDrNtidle = TDrN / rValues[inn] + Tppdr_temp;
            
            if (NrValues[inn] == 0) {
                DDr_logN_div_logN  = 0.0;
                DDr_m_Dr_dr_N_div_logN = 0.;
            }
            if (std::isnan(DrN))  DrN = 0.0;
            if (std::isnan(DDrN))  DDrN = 0.0;
            if (std::isnan(DDr_logN_div_logN))  DDr_logN_div_logN = 0.0;
            if (std::isnan(DDrNtidle))  DDrNtidle = 0.0;
            if (std::isnan(DDr_m_Dr_dr_N_div_logN))  DDr_m_Dr_dr_N_div_logN = 0.0;
            outputFile << iy * 0.2 << "  " << rValues[inn] << "  " << DrN << "  " << DDrN << "  " << DDr_logN_div_logN << "  " << DDrNtidle 
                       << "  " << DDr_m_Dr_dr_N_div_logN << endl;
        }
        gsl_interp_free(interp2);
        gsl_interp_accel_free(acc);
    }
    outputFile.close();
}

