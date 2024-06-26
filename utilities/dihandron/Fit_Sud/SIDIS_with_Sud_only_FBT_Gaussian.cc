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
LHAPDF::PDF* FF;
LHAPDF::PDF* pdf;

using namespace std;

int main(int argc, char* argv[]) {

        std::stringstream filename;
    filename << "alphas_Q2" ;

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
    
    int length = 10000;
    for (int itheta=100 ; itheta<length +1; itheta++) {
        double Q2 = itheta * 0.01;
        realA << Q2 << "  ";
        realA << pdf->alphasQ2(Q2);// call from LHAPDF
        realA << endl;
    }
    
    realA.close();
    return 0;
}

