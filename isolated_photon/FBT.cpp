//###############################################################################
//#                                                                             #
//#                Fast Bessel Transform (FBT) for TMDs                         #
//#     Zhongbo Kang, Alexei Prokudin, Nobuo Sato, John Terry                   #
//#                   Please cite ArXiv:1906.05949                              #
//#                  N is number of function calls                              #
//#                  nu is Bessel function order                                #
//###############################################################################
#define _USE_MATH_DEFINES // using M_PI for pi

#include <cmath> // abs
#include <math.h>
#include <stdio.h>
#include <iostream>
#include "FBT.h"

//acknowledgement
void FBT::acknowledgement(){
    std::cout << "###############################################################################" << std::endl;
    std::cout << "#                                                                             #" << std::endl;
    std::cout << "#                Fast Bessel Transform (FBT) for TMDs                         #" << std::endl;
    std::cout << "#     Zhongbo Kang, Alexei Prokudin, Nobuo Sato, John Terry                   #" << std::endl;
    std::cout << "#                   Please cite Kang:2019ctl                                  #" << std::endl;
    std::cout << "#                  nu is Bessel function order                                #" << std::endl;
    std::cout << "#                  option = 0,1,2  (modified, unmodified, fixed h) Ogata      #" << std::endl;
    std::cout << "#                  N is number of function calls                              #" << std::endl;
    std::cout << "#                  Q initial guess where function has maximum                 #" << std::endl;
    std::cout << "###############################################################################" << std::endl;
};

//citation
void FBT::citation(){
    std::cout << "###############################################################################" << std::endl;
    std::cout << "#                     Thank you for using FBT!                                #" << std::endl;
    std::cout << "# Please, cite Kang:2019ctl if used for a publication                         #" << std::endl;
    std::cout << "###############################################################################" << std::endl;
};

// Deconstructor
FBT::~FBT(){
  //jn_zeros0.~vector<double>();
  //citation();
};

// Constructor
FBT::FBT(double _nu, int _option, int _N, double _Q){
  if( _nu >= 0.){
    this->nu     = _nu;
  } else {
    std::cerr << " The value of nu = " << _nu << " is not supported." << std::endl;
    std::cerr << " Falling back to default  nu = " << FBT::nu_def << std::endl;
    this->nu     = FBT::nu_def;
  }

  if( _N >= 1){
    this->N     = _N;
  } else {
    std::cerr << " The value of N = " << _N << " is not supported." << std::endl;
    std::cerr << " Falling back to default  N = "  << FBT::N_def <<std::endl;
    this->N     = FBT::N_def;
  }


  if( _Q > 0){
    this->Q     = _Q;
  } else {
    std::cerr << " The value of Q = " << _Q << " is not supported." << std::endl;
    std::cerr << " Falling back to default  Q = "  << FBT::Q_def <<std::endl;
    this->Q     = FBT::Q_def;
  }

  if( _option <= 2 && _option >= 0){
    this->option     = _option;
  } else {
    std::cerr << " The value of option = " << _option << " is not supported." << std::endl;
    std::cerr << " Falling back to default  option = "  << FBT::option_def <<std::endl;
    this->option     = FBT::option_def;
  }

  // Sets maximum number of nodes to about 2^15:
  const int maxN = 32769;

  //Imports zeros of the Bessel function. Initializing this way speeds up calls
  try
  {
    boost::math::cyl_bessel_j_zero(this->nu, 1, maxN, std::back_inserter(jn_zeros0));
    for (size_t i = 0; i < maxN; i++) {
      zeros.push_back( jn_zeros0[i] );
      xi.push_back( zeros[i]/M_PI );
      Jp1.push_back( boost::math::cyl_bessel_j(nu+1.,M_PI*xi[i]) ); //The functions gsl_sf_bessel_Jn and gsl_sf_bessel_Yn return the result of the Bessel functions of the first and second kinds respectively
      w.push_back( boost::math::cyl_neumann(nu,M_PI*xi[i])/Jp1[i] );  
    }
  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

  //acknowledgement();

};

double get_psi( double t){
  return ( t )*tanh( M_PI/2.* sinh( t ) );
};

double get_psip( double t){
  return M_PI*t*( -pow(tanh( M_PI*sinh(t)/2.),2) + 1.)*cosh(t)/2. + tanh(M_PI*sinh(t)/2.);
};


//double f_for_ogata
double f_for_ogata(double x, std::function<double (double) > g, double q){
  return g(x/q)/q;
};

//Transformed Ogata quadrature sum. Equation ? in the reference.
double FBT::ogatat(std::function<double (double) > f, double q, double h){
  double nu = this->nu;
  int N = this->N;

  double knots, Jnu, psip, F;

  double val = 0;

  try{
    for (size_t i = 0; i < (unsigned)N; i++) {
      knots = M_PI/h*get_psi( h*xi[i] );
      Jnu = boost::math::cyl_bessel_j(nu,knots);
      double temp =  get_psip( h*xi[i] );

      if( isnan(temp) )
      {
        psip = 1.;
      }
      else
      {
        psip = temp;
      };
      F = f_for_ogata(knots, f, q);
      val += M_PI*w[i]*F*Jnu*psip;


    }
  }
  catch (std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

  return val;
};

//"""Untransformed Ogata quadrature sum. Equation ? in the reference."""
double FBT::ogatau(std::function<double (double) > f, double q, double h){
  double nu = this->nu;
  int N = this->N;


  double knots, F;

  double val = 0;

  try
  {
    for (size_t i = 0; i < (unsigned)N; i++) {
      knots = xi[i]*h;
      F=f_for_ogata(knots, f, q)*boost::math::cyl_bessel_j(nu,knots);
      val+=h*w[i]*F;
    }
  }
  catch(std::exception& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

  return val;
};

//double f_for_get_hu
double f_for_get_hu(double x, std::function<double (double) > g, double q){
   return -abs(x*g(x/q));
};

//"""Determines the untransformed hu by maximizing contribution to first node. Equation ? in ref."""
double FBT::get_hu(std::function<double (double) > f, double q){
  double Q = this->Q;

  double zero1 = jn_zeros0[0];
  const int double_bits = std::numeric_limits<double>::digits;
  std::pair<double, double> r = boost::math::tools::brent_find_minima(std::bind(f_for_get_hu, std::placeholders::_1, f, q), Q/10., 10.*Q, double_bits);

  double hu = r.first/zero1*M_PI;
  if(hu >= 2.){
    hu = 2.;
//    std::cerr<< "Warning: Number of nodes is too small N = " << this->N << std::endl;
  }

  return hu;
};


//"""Determine transformed ht from untransformed hu. Equation ? in ref."""
double FBT::get_ht(double hu){
  int N = this->N;

  double zeroN = double(jn_zeros0[N-1]);

  return M_PI/zeroN*asinh(2./M_PI*atanh(hu/M_PI));
};


//"""Transformed optimized Ogata."""
double FBT::fbt(std::function<double (double) > g,  double q){
  /* Numerical computation of F(q)=int(f(x)*Jn(x*q)) Bessel transform
		[in]g     - integrand
		[in]q - evaluation point

	return:
			-computed integral value
	*/

  double hu = 0.0;
  double ht = 0.0;
  double result = 0.0;

  if (this->option == 0){ // default modified Ogata
    hu = get_hu(g,q);
    ht = get_ht(hu);
    result = ogatat(g,q,ht);
  } else if (this->option == 1){ // umodified Ogata
    hu = get_hu(g ,q);
    result = ogatau(g,q,hu);
  } else if (this->option == 2){ // modified Ogata h = 0.05
    hu = 0.05;
    result = ogatat(g,q,hu);
  };

  return result;

};
