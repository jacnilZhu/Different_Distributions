#include "stdafx.h"
#include <random>
#include "FractureNetwork.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm> 
#include<math.h>
#include "amp_math.h"
#include "MersenneTwister.h"
#include <map>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/QR>
#include "myrandom.h"
#include<complex>
using namespace std;

// the inverse of the error function is not trivial and thanks to the answer on stack overflow;
// https://stackoverflow.com/questions/27229371/inverse-error-function-in-c?lq=1

float my_logf(float);
/* compute inverse error functions with maximum error of 2.35793 ulp */
float erfinv(float a)
{
	float p, r, t;
	t = fmaf(a, 0.0f - a, 1.0f);
	t = my_logf(t);
	if (fabsf(t) > 6.125f) { // maximum ulp error = 2.35793
		p = 3.03697567e-10f; //  0x1.4deb44p-32 
		p = fmaf(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26 
		p = fmaf(p, t, 1.22150334e-6f); //  0x1.47e512p-20 
		p = fmaf(p, t, 2.84108955e-5f); //  0x1.dca7dep-16 
		p = fmaf(p, t, 3.93552968e-4f); //  0x1.9cab92p-12 
		p = fmaf(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9 
		p = fmaf(p, t, 4.83185798e-3f); //  0x1.3ca920p-8 
		p = fmaf(p, t, -2.64646143e-1f); // -0x1.0eff66p-2 
		p = fmaf(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1 
	}
	else { // maximum ulp error = 2.35456
		p = 5.43877832e-9f;  //  0x1.75c000p-28 
		p = fmaf(p, t, 1.43286059e-7f); //  0x1.33b458p-23 
		p = fmaf(p, t, 1.22775396e-6f); //  0x1.49929cp-20 
		p = fmaf(p, t, 1.12962631e-7f); //  0x1.e52bbap-24 
		p = fmaf(p, t, -5.61531961e-5f); // -0x1.d70c12p-15 
		p = fmaf(p, t, -1.47697705e-4f); // -0x1.35be9ap-13 
		p = fmaf(p, t, 2.31468701e-3f); //  0x1.2f6402p-9 
		p = fmaf(p, t, 1.15392562e-2f); //  0x1.7a1e4cp-7 
		p = fmaf(p, t, -2.32015476e-1f); // -0x1.db2aeep-3 
		p = fmaf(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1 
	}
	r = a * p;
	return r;
}

/* compute natural logarithm with a maximum error of 0.85089 ulp */
float my_logf(float a)
{
	float i, m, r, s, t;
	int e;

	m = frexpf(a, &e);
	if (m < 0.666666667f) { // 0x1.555556p-1
		m = m + m;
		e = e - 1;
	}
	i = (float)e;
	/* m in [2/3, 4/3] */
	m = m - 1.0f;
	s = m * m;
	/* Compute log1p(m) for m in [-1/3, 1/3] */
	r = -0.130310059f;  // -0x1.0ae000p-3
	t = 0.140869141f;  //  0x1.208000p-3
	r = fmaf(r, s, -0.121484190f); // -0x1.f19968p-4
	t = fmaf(t, s, 0.139814854f); //  0x1.1e5740p-3
	r = fmaf(r, s, -0.166846052f); // -0x1.55b362p-3
	t = fmaf(t, s, 0.200120345f); //  0x1.99d8b2p-3
	r = fmaf(r, s, -0.249996200f); // -0x1.fffe02p-3
	r = fmaf(t, m, r);
	r = fmaf(r, m, 0.333331972f); //  0x1.5554fap-2
	r = fmaf(r, m, -0.500000000f); // -0x1.000000p-1
	r = fmaf(r, s, m);
	r = fmaf(i, 0.693147182f, r); //  0x1.62e430p-1 // log(2)
	if (!((a > 0.0f) && (a <= 3.40282346e+38f))) { // 0x1.fffffep+127
		r = a + a;  // silence NaNs if necessary
					//if (a  < 0.0f) r = (0.0f / 0.0f); //  NaN
					//if (a == 0.0f) r = (-1.0f / 0.0f); // -Inf
	}
	return r;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double Cum_lognormal(double x, double mu, double sigma2)
{
	double sigma = sqrt(sigma2);
	double phi = 0.5*(1 + erf((log(x) - mu) / (sigma*sqrt(2))));
	return phi;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::normal(double mean, double stddev)
{
	double sigma = stddev;
	double middle1 = randomReal(0, 1, id, a) ;
	double middle2 = middle1 * 2 - 1;
	double result = erfinv(middle2)*sigma*sqrt(2) + mean;
	return result;


}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::normalCDF(double value, double mean, double var)
{
	double sigma = sqrt(var);
	double result = 0.5*(1.0 + erf((value - mean) / (sigma*sqrt(2))));
	return result;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::Truncatednormal(double mean, double var, double lower, double upper)
{
	double sigma = sqrt(var);
	double Phi_upper = normalCDF(upper, mean, var);
	double Phi_lower = normalCDF(lower, mean, var);
	double middle1 = (Phi_upper - Phi_lower)*randomReal(0, 1, id, a) + Phi_lower;
	double middle2 = middle1 * 2.0 - 1.0;
	double middle3 = erfinv(middle2)*sigma*sqrt(2) + mean;
	return middle3;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
 
double FractalNetwork::lognormal(double mean, double var)
{
	double a = mean; double b = var;// a is the mean of the lognormal variable, b is the variance.
	double sigma2 = log(1 + b / a / a);
	double sigma = sqrt(sigma2);
	double mu = log(a) - 0.5*sigma2;
	double result = exp(erfinv(2 * randomReal(0, 1, 1, 1) - 1)*sigma*sqrt(2) + mu);
	return result;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::Truncatedlognormal(double mean, double var,double lower, double upper)
{
	double a = mean; double b = var;// a is the mean of the lognormal variable, b is the variance.
	double sigma2 = log(1.0 + b / a / a);
	double sigma = sqrt(sigma2);
	double mu = log(a / (sqrt(1.0 + b / a / a)));
	double Phi_upper = Cum_lognormal(upper, mu, sigma2);
	double Phi_lower= Cum_lognormal(lower, mu, sigma2);
	double middle1 = ((Phi_upper - Phi_lower)*randomReal(0, 1, id, a) + Phi_lower) * 2.0 - 1.0;
	double middle2 = erfinv(middle1)*sigma*sqrt(2) + mu;
	double result = exp(middle2);
	return result;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::Exponential(double lambda)
{
	double result = log(1.0 - randomReal(0, 1, id, a)) / -lambda;
	return result;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//

double Cum_exponential(double value, double lambda)
{
	double result = 1.0 - exp(-lambda * value);
	return result;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::Truncatedexponential(double lambda, double lower, double upper)
{
	double Phi_upper = Cum_exponential(upper, lambda);
	double Phi_lower = Cum_exponential(lower, lambda);
	double middle1 = (Phi_upper - Phi_lower)*randomReal(0, 1, id, a) + Phi_lower;
	double result = log(1.0 - middle1) / -lambda;
	return result;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::Powerlaw(double lmax, double lmin, double a)
{
	double c = pow(lmax, 1 - a) - pow(lmin, 1 - a);
	double result = (pow((randomReal(0, 1, 1, a)*c + pow(lmin, 1 - a)), (1 / (1 - a))));
	return result;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double  FractalNetwork::Gammalaw(double k, double alpha)// k is the scale parameter and alpha is the shape parameter
{	// first we generate variable following gamma(1,alpha);
	// since the scaling characteristic of the gamma law, kX~gamma(k,alpha);
	// we will return the result k*X;
	// The programm is based on two papers:
	// (alpha>1) Extremely efficient generation of Gamma random variables for$\backslash$alpha>= 1
	// (alpha<=1) A convenient way of generating gamma random variables using generalized exponential distribution
	double result;
	if (alpha <= 1)
	{
		double d = 1.0334 - 0.0766*pow(exp(1), 2.2942*alpha);
		double a = pow(2.0, alpha)*pow((1 - pow(exp(1), -d / 2.0)), alpha);
		double b = alpha * pow(d, alpha - 1.0)*pow(exp(1), -d);
		double c = a + b;
		double U, X, V, test1, test2;
		while (true)
		{
			U = randomReal(0, 1, 0, 0);
			if (U <= a / (a + b))
			{
				X = -2 * log(1 - pow(c*U, 1.0 / alpha) / 2.0);
			}
			else
			{
				X = -log(c*(1 - U) / (alpha*pow(d, alpha - 1.0)));
			}
			V = randomReal(0, 1, 0, 0);
			if (X <= d)
			{
				test1 = pow(X, alpha - 1.0)*pow(exp(1), -X / 2.0) / pow(2, alpha - 1.0) / pow((1 - pow(exp(1), -X / 2.0)), alpha - 1.0);
				if (V <= test1)
				{
					result = X;
					break;
				}
			}
			else
			{
				test2 = pow(d / X, 1.0 - alpha);
				if (V <= test2)
				{
					result = X;
					break;
				}
			}
		}
		return k * result;
	}
	else
	{
		k = 1.0 / k;// convert to the rate parameter
		int alphap = int(alpha);// integer part of alpha
		double betap, kp, X, p0x, p1x; double U = 1;
		if (alpha >= 2)
		{
			betap = k * (alphap - 1.0) / (alpha - 1.0);
			kp = exp(alphap - alpha)*(pow((alpha - 1.0) / k, alpha - alphap));
		}
		else if (alpha < 2 && alpha>1)
		{
			betap = k / alpha;
			kp = exp(1.0 - alpha)*pow((alpha / k), alpha - 1.0);
		}
		while (1)
		{
			for (int i = 0; i < alphap; i++)
			{
				U = U * randomReal(0, 1, 0, 0);
			}
			X = -log(U) / betap;
			U = randomReal(0, 1, 0, 0);
			p0x = pow(X, alpha - 1.0)*exp(-k * X);
			p1x = kp * pow(X, alphap - 1.0)*exp(-betap * X);
			if (U <= p0x / p1x)
			{
				result = X;
				break;
			}
		}

		return  result;
	}


}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
// von Mises-Fisher distribution on a unit circle;
double FractalNetwork::FisherDis(double theta, double  kappa)
{   // theta is  preferred direction
	// kappa is the concentration parameter;
	// This code is modified based on the function, circ_vmrnd, in a Matlab toolbox;
	// CircStat: a MATLAB toolbox for circular statistics
	// Berens, Philipp and others

	

	double alpha = 0; double *u; u = new double[3];
	double a, b, r, z, f, c;
	// if kappa is too small, it is just a uniform distribution;
	if (kappa < 1e-6)
	{
		alpha = 2.0 * PI*randomReal(0, 1,0,1);
		return alpha;
	}

	a = 1 + sqrt((1.0 + 4.0 * pow(kappa, 2.0)));
	b = (a - sqrt(2.0 * a)) / (2.0 * kappa);
	r = (1 + pow(b, 2)) / (2 * b);


	while (true)
	{
		u[0] = randomReal(0, 1,0,1);
		u[1] = randomReal(0, 1,0,1);
		u[2] = randomReal(0, 1,0,1);
		z = cos(PI*u[0]);
		f = (1.0 + r * z) / (r + z);
		c = kappa * (r - f);

		if (u[1] < c * (2.0 - c) || !(log(c) - log(u[1]) + 1.0 - c < 0))
			break;
	}


	alpha = theta + sign(u[2] - 0.5) * acos(f);
	alpha = angle(exp(1i*alpha));
	return alpha;



}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
// von Mises-Fisher distribution on a unit sphere
// The method is adopted from the paper:
// G. Kurz and U. D. Hanebeck. Stochastic sampling of the hyperspherical von Mises-Fisher distribution without rejection methods. In 2015 Sensor Data  
// Fusion: Trends, Solutions, Applications (SDF), pages 1{6. IEEE, 2015.
pointcor3D FractalNetwork::VMFisherDis(pointcor3D mu, double kappa)// this is the von-mises fisher distirbution on a 3D sphere
{
	pointcor3D result;
	double w; 
	double u = randomReal(0, 1, 0, 0);
	pointcor v;// the 2D vector (x1,x2),follows a bivariate Gaussian with mean zero and identity covariance, which means x1 and x2 are independent
	//w = 1 + 1.0 / kappa * (log(u + (1.0 - u)*exp(-2 * kappa)));
	w = 1.0 / kappa * log(2 * u*sinh(kappa) + exp(-kappa));
	v.xcor = normal(0, 1);
	v.ycor = normal(0, 1);
	double dis = pow(v.xcor*v.xcor + v.ycor*v.ycor, 0.5);
	// normalize v;
	v.xcor = v.xcor / dis;
	v.ycor = v.ycor / dis;
	Vector3f pt;
	pt(0) = w;
	pt(1) = pow((1.0 - w * w), 0.5)*v.xcor;
	pt(2) = pow((1.0 - w * w), 0.5)*v.ycor;
	// rotate the pt w.r.t mu
	MatrixXf m(3, 3);
	MatrixXf Q(3, 3);
	MatrixXf R(3, 3);
	for(int i=0;i<3;i++)
		for (int j = 0; j < 3; j++)
		{
			m(i, j) = 0;
		}
	m(0, 0) = mu.xcor;
	m(1, 0) = mu.ycor;
	m(2, 0) = mu.zcor;
	HouseholderQR<MatrixXf> qr(m);
	Q = qr.householderQ();
	R = qr.matrixQR().triangularView<Upper>();
	if (R(0, 0) < 0)
		Q = -Q;
	pt = Q * pt;// rotate pt 
	result.xcor = pt(0);
	result.ycor = pt(1);
	result.zcor = pt(2);
	return result;
 
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::sign(double x)
{
	if (x > 0)
		return 1;
	else if (x < 0)
		return -1;
	else
		return 0;


}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
double FractalNetwork::angle(complex<double> h)
{
	double result = atan2(imag(h), real(h));
	return result;
}
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//


 

