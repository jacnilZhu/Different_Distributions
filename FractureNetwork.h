#ifndef FRACTURENETWORK_H_INCLUDED
#define FRACTURENETWORK_H_INCLUDED
#include <vector>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat; //declear a sparse matrix;
typedef Eigen::SparseVector<double> SpVec;// declear a sparse vector;




struct pointcor
{
	double xcor;
	double ycor;
};
struct pointcor3D
{
	double xcor;
	double ycor;
	double zcor;
};
//*******************************************************


//*******************************************************


//*******************************************************




//*******************************************************
class FractalNetwork
{
public:
 
	FractalNetwork() {} // constructor: initialization
	~FractalNetwork() {}  //deconstructor
	// variables
	const double PI = 3.14159265359;
	const double tol = 1e-10;
	const double NN = 99999999999;
 	int id=1;
	double a=3.0;

	// different distribution
	double lognormal(double mean, double var);
	double normal(double mean, double stddev);
	double fisher(double mean, double stddev);
	
	
	double FisherDis(double theta, double  kappa);
	double sign(double x);
	double FractalNetwork::angle(complex<double> h);

 	void CurveFitting(string InputDataname, int nspline, int N);
	void MLPNN(string inputdataname,double min,double max);
	void Generate_fitting_variable(double min, double max, string filename);
	void  Outputarray(string name, int num, double *array);
	int inputFittingData(string name, double* X, double*Y);



	double  Gammalaw(double k, double alpha);
	pointcor3D VMFisherDis(pointcor3D mu, double kappa);
	double area(int start, int end, double *X, double *Y);
	double FractalNetwork::splayangle();
	double FractalNetwork::splayangle(double *cdf, double *X, int DataNum);
	double FractalNetwork::average(double *arr, int num);
	int FractalNetwork::FindIndex(double *vec, double x, int start, int end);
	double FractalNetwork::Truncatedlognormal(double mean, double var, double lower, double upper);
	double FractalNetwork::normalCDF(double value, double mean, double var);
	double FractalNetwork::normalCDF(double value);
	double FractalNetwork::Exponential(double lambda);

	double FractalNetwork::Truncatednormal(double mean, double var, double lower, double upper);
	double FractalNetwork::Truncatedexponential(double lambda, double lower, double upper);
	double FractalNetwork::Powerlaw(double lmax, double lmin, double a);
	void FractalNetwork::TestDifferentDis();

};

#endif // FRACTALNETWORK_H_INCLUDED


