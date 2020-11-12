#include "stdafx.h"
#include "FractureNetwork.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <ctime>
#include <fstream>
#include <random>
#include <iomanip>
#include <string>
#include <algorithm> 
#include<ctime>
#include "MersenneTwister.h"
using namespace std;
int main()
{
	FractalNetwork mynetwork;// define class object
	// These two lines are used to run a MLP calculation and generate variables following the sampling distribution;
	mynetwork.MLPNN("Data.txt", 75.1, 328.4013);// MLP neural network
  	mynetwork.Generate_fitting_variable(1,2,"variable.txt");
	// This line is used to test all stochastic distributions in the paper;
	mynetwork.TestDifferentDis();
	cout << "finished" << endl;
	system("pause");
	return 0;

}




