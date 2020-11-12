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
 







void  FractalNetwork::Outputarray(string name,int num,double *array)
{


	ofstream output1;

	output1.open(name);
	output1.setf(ios::showpoint);
	output1.precision(8);
	for (int i = 0; i < num ; i++)
	{

		output1 << array[i] << endl;
	}

}
