// bessel.cpp
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>
#include "FBT.h"


double test( double x, double width ){ return x*exp(-x/width);} // test function to transform data allows to send anything else to the function
double exact( double qT){ return pow((1.+qT*qT),-1.5);} // test function exact


double test1( double x){ return x*exp(-x*x);} // test function to transform
double exact1( double qT ){ return exp(-qT*qT/4.)/2.;} // test function exact

double test2( double x, double A, double B, double C, double D, double E, double F ) { 
       return A * exp(-pow(x, B) * C + D) + E * pow(x, F);} // test function to transform data allows to send anything else to the function

double interp_r( double r, std::vector<double>& xValues, std::vector<double>& yValues) { 
    const size_t numPoints = 400;
    // Create a linear interpolation object
    gsl_interp *interp = gsl_interp_alloc(gsl_interp_linear, numPoints);
    gsl_interp_init(interp, xValues.data(), yValues.data(), numPoints);
    double interpolatedValue = 0.0;
    if (r < 100) interpolatedValue = gsl_interp_eval(interp, xValues.data(), yValues.data(), r, nullptr);
    return interpolatedValue * r * 2. * M_PI;
}

    
using namespace std;
int main( void )
{
    FBT ogata0 = FBT(); // Fourier Transform with Jnu, nu=0.0 and N=10
    // Open the input file
    std::ifstream inputFile("rcBK_MVe_Heikki_table.txt");
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return 1;
    }

    // Read data from the file
    std::vector<double> YValues, rValues, NrValues, Integrated_NrValues;
    double Y, r, Nr; // r in [GeV^-1]
    while (inputFile >> Y >> r >> Nr) {
        YValues.push_back(Y);
        rValues.push_back(r);
        NrValues.push_back(1. - Nr);
    }
    // Close the file
    inputFile.close();
    
    // read in integrated Nr
    std::ifstream inputFileNr("integrated_Nr");
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
    for (auto iy = 0; iy < 81; iy++) {
        for (int ii =0; ii < 2000; ii ++) {
            double qT = ii * 0.005;
            outputFile << iy * 0.2 << "  " << qT << "  ";
            // Create a subset vector using iterators
            auto subsetBeginr = rValues.begin() + iy * 400;
            auto subsetEndr = rValues.begin() + iy * 400 + 400;
            std::vector<double> subsetrVector(subsetBeginr, subsetEndr);
            
            auto subsetBegin = NrValues.begin() + iy * 400;
            auto subsetEnd = NrValues.begin() + iy * 400 + 400;
            std::vector<double> subsetNrVector(subsetBegin, subsetEnd);
            double res = 2. * M_PI * ogata0.fbt(std::bind(interp_r, std::placeholders::_1, 
                         subsetrVector, subsetNrVector),qT);
            if (ii ==0) {
                res = Integrated_NrValues[iy];
            }
            outputFile << res << endl;
        }
    }
    outputFile.close();
}
