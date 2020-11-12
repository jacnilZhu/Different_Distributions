#include "stdafx.h"
#include "FractureNetwork.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm> 
#include "MersenneTwister.h"
#include "myrandom.h"
#include <map>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
using namespace std;
using namespace Eigen;
void ouputdis3D(int num, string name, pointcor3D *data);
void ouputdis(int num, string name, double *data);

 
void FractalNetwork::TestDifferentDis()
{
	// test normal distribution
	int num = 100000;
	double *data; data = new double[num];
	pointcor3D *pts; pts = new pointcor3D[num];
	for (int i = 0; i <num; i++)
	{
		// provide the mean and variance for stochastic distribution;
 		double mu = 0; double var = 0.5;
		var = var*var;
		double a = exp(mu + 0.5*var);
		double b = (exp(var)-1)*(exp(mu*2+var));
		double lower = 0.8*a; double higher = 1.2*a;// we need to set the lower and upper limits for the truncated lognormal distribution.
		//data[i] = Truncatedlognormal(a,b,lower, higher);
		//data[i] = lognormal(a,b);
		//data[i] = FisherDis(0, 4);
		//data[i] = Truncatednormal(0,1,-3,2);
		//data[i] = Truncatedexponential(1,1,3);
		//data[i] = Gammalaw(1,0.5);
		//data[i] = Exponential(2);
		
		// 3D von Mises-Fisher distribution; activate the following two lines;
		//pointcor3D pt; pt.xcor = 1; pt.ycor = 0; pt.zcor = 0;
		//pts[i] = VMFisherDis(pt, 10);

	}
	//output 2D distribution data;
	//ouputdis(num, "normal.txt", data);
	//output 3D distribution data;
	//ouputdis3D(num, "normal.txt", pts);
	delete[] data;
	delete[] pts;
}



void ouputdis(int num, string name, double *data)
{
	ofstream output;

	output.open(name);
	output.setf(ios::showpoint);
	output.precision(8);
	for (int i = 0; i < num; i++)
	{
			output << data[i] << endl;
	}
}

void ouputdis3D(int num, string name, pointcor3D *data)
{
	ofstream output;

	output.open(name);
	output.setf(ios::showpoint);
	output.precision(8);
	for (int i = 0; i < num; i++)
	{
		output << data[i].xcor<<"	"<<data[i].ycor<<"	"<<data[i].zcor << endl;
	}
}