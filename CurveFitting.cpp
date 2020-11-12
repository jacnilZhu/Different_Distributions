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
#include "MersenneTwister.h"
#include "myrandom.h"
#include <map>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
using namespace std;
using namespace Eigen;


//#define 
int N = 5;
#define epsilon 0.005
#define epoch 50000
#define mu 0.1
 

double sigmoid(double x)
{
	return (1.0 / (1.0 + exp(-x)));
}

double activation(double x,int i)
{
	switch (i)
	{
	case 1: //sigmoid;
	{
		return (1.0 / (1.0 + exp(-x)));
	}
	case 2: // tanh
	{
		return (exp(1.0*x)- exp(-x)) / (exp(1.0*x) + exp(-x));
	}
	}
}


double gradient_of_activation(double x, int i)
{
	switch (i)
	{
	case 1: //sigmoid;
	{
		return (activation(x,i)*(1.0-activation(x,i)));
	}
	case 2: // tanh
	{
		return (1-activation(x,i)*activation(x,i));
	}
	}
}

double f_theta(double x, int N,double *c,double *W,double *V,double b)
{
	double result = b;
	for (int i = 0; i < N; i++) {
		result += V[i] * activation(c[i] + W[i] * x,1);
	}
	return result;
}


void train(double x, double y,int N, double *c, double *W, double *V, double b)
{
	for (int i = 0; i < N; i++) {
		W[i] = W[i] - epsilon * 2 * (f_theta(x,N,c,W,V,b) - y) * V[i] * x *
			(1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
	}
	for (int i = 0; i < N; i++) {
		V[i] = V[i] - epsilon * 2 * (f_theta(x, N, c, W, V, b) - y) * sigmoid(c[i] + W[i] * x);
	}
	b = b - epsilon * 2 * (f_theta(x, N, c, W, V, b) - y);
	for (int i = 0; i < N; i++) {
		c[i] = c[i] - epsilon * 2 * (f_theta(x, N, c, W, V, b) - y) * V[i] *
			(1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
	}
}


int FractalNetwork::inputFittingData(string name,  double* X, double*Y)
{
	ifstream inputdata;
	inputdata.open(name);
	int i = 0;
 
	while (!inputdata.eof())
	{
		inputdata >> X[i]>>Y[i];
		i++;
	}
	return i - 1;
}


void  OutputData(string name,int N,double *X,double*Y)
{
	ofstream output;

	output.open(name);
	output.setf(ios::showpoint);
	output.precision(8);
	for (int i = 0; i < N; i++)
	{
		 

			output <<X[i]<<"	"<<Y[i]<< endl;
		 
	}

}

// this is a simple 2-layer machining learning algorithm, not suitable for complex discrete sampling distributions.
void FractalNetwork::CurveFitting(string InputDataname, int nspline, int N)
{

	double *X; double* Y; double *W; double *V; double *c; double b = 0; int capacity = 5000;
	X = new double[capacity]; Y = new double[capacity];
	W = new double[N]; V = new double[N]; c = new double[N];

	int DataNum;
	// initialization of parameters
	for (int i = 0; i < N; i++) {
		W[i] = randomReal(0, 1, id, a);
		V[i] = randomReal(0, 1, id, a);
		c[i] = randomReal(0, 1, id, a);
	}

	DataNum = inputFittingData(InputDataname, X, Y);
	cout << "data num is " << DataNum << endl;
	//system("pause");
	// update the weight
	for (int j = 0; j < epoch; j++) {
		for (int i = 0; i < DataNum; i++) {
			train(X[i], Y[i],N,c,W,V,b);
		} 
		//cout << j << endl;
	}

	double value;
	for (int i = 0; i < 1000; i++)
	{
		X[i] = i * 2*PI/ 1000;
		Y[i] = f_theta(X[i], N, c, W, V, b);

	}
	OutputData("fitting.txt",1000, X, Y);

	delete[] W; delete[] V; delete[] c; delete[] X; delete[] Y;
}




struct	Neuron 
{ 
	double weight[20];
	double previousweight[20];
	double outputvalue;// after the activation function;
	double inputvalue;// before the activation function;
	double delta;
	double bias;
	double weightadjustment[20];
	double biasadjustment;
 
};



void BackPropagation(Neuron ** Layer, int layernum, int*nodenum, double *truth,int activationnum, double learningrate)
{
 
	// calculate the delta_i for each node in each layer, back-propagation;
	for (int i = 0; i < nodenum[layernum - 1]; i++)// for the output layer, the delta is simply the difference between predicted value and the true value
	{
		Layer[layernum - 1][i].delta = Layer[layernum - 1][i].outputvalue - truth[i];
	}
	for (int i = layernum - 2; i > 0; i--)// for all other layers from layer 2 to layer n-1, calculate delta for each node in a backward direction
	{
		for (int j = 0; j < nodenum[i]; j++)// go through all the nodes in the layer i
		{
			for (int k = 0; k < nodenum[i + 1]; k++)// for each node in the next layer;
			{
				Layer[i][j].delta = Layer[i + 1][k].delta*Layer[i+1][k].weight[j];
			}
			Layer[i][j].delta = gradient_of_activation(Layer[i][j].inputvalue, activationnum)*Layer[i][j].delta;
		}
			
	}

	// find the weight adjustment

	for (int i = 1; i < layernum; i++)
	{
		for (int j = 0; j < nodenum[i]; j++)
		{
			Layer[i][j].biasadjustment=  learningrate*(Layer[i][j].delta);// each node has a bias;
																				 // update the weight w=w-e*gradient(w);
			for (int k = 0; k < nodenum[i - 1]; k++)// each node has a array of weights for each node in the previous layer
			{
				Layer[i][j].weightadjustment[k] =  learningrate*Layer[i][j].delta*Layer[i - 1][k].outputvalue;
				//Layer[i][j].weightadjustment[k] = learningrate*(mu*(Layer[i][j].weight[k]- Layer[i][j].previousweight[k])+(1-mu)*Layer[i][j].delta*Layer[i - 1][k].outputvalue);

			}
		}

	}

}

void update_weigth_bias(Neuron **Layer, int layernum, int *nodenum, int activationnum,double learningrate)
{


	// updata weights and bias in each node except the input layer nodes by descent gradient method;
	for (int i = 1; i < layernum ; i++)
	{
		for (int j = 0; j < nodenum[i]; j++) 
		{
			Layer[i][j].bias = Layer[i][j].bias - Layer[i][j].biasadjustment;// each node has a bias;
			Layer[i][j].biasadjustment = 0;
			// update the weight w=w-e*gradient(w);
			for (int k = 0; k < nodenum[i - 1]; k++)// each node has a array of weights for each node in the previous layer
			{
				Layer[i][j].previousweight[k] = Layer[i][j].weight[k];
				Layer[i][j].weight[k] = Layer[i][j].weight[k] - Layer[i][j].weightadjustment[k];
				Layer[i][j].weightadjustment[k] = 0;
 
			}
		}

	}

	 
 
}


void ForwordPropagation(Neuron **Layer, int layernum, int* nodenum,int activationnum)
{
	double inputvalue = 0; double outputvalue;
	for (int i = 1; i < layernum; i++)// except the input layer, we need to calculate all the outputs in the hidden layer and the output layer.
	{
		for (int j = 0; j < nodenum[i]; j++)// for each node in the layer i, we need to calculate the outputvalue;
		{
		 
			for (int k = 0; k < nodenum[i - 1]; k++)// summation of the values in the previous layer times the corresponding weights + the bias
			{
				inputvalue = inputvalue + Layer[i][j].weight[k] * Layer[i - 1][k].outputvalue;

			}
			inputvalue += Layer[i][j].bias;
			Layer[i][j].inputvalue = inputvalue;
			Layer[i][j].outputvalue = activation(inputvalue, activationnum);
			 
		}

	}
}

void initializeLayer(Neuron **Layer, int layernum, int* nodenum)
{
	for (int i = 1; i < layernum; i++)// all layers except the input layer have weights;
	{
		for (int j = 0; j < nodenum[i]; j++)// assign weight to each node 
		{

			// assign weights to current nodes
			for (int k = 0; k < nodenum[i - 1]; k++)// each node has a array of weights for each node in the previous layer
			{
				double r = 1.0 / sqrt(nodenum[i - 1]);
				Layer[i][j].weight[k] = randomReal(-r, r, 1, 1);
				//Layer[i][j].weight[k] = randomReal(-1, 1, 1, 1);
				Layer[i][j].previousweight[k] = Layer[i][j].weight[k];
				Layer[i][j].weightadjustment[k] = 0;
				Layer[i][j].bias = randomReal(-r, r, 1, 1);// each node has a bias;
				//Layer[i][j].bias = randomReal(-1, 1, 1, 1);// each node has a bias;
				Layer[i][j].biasadjustment = 0;
			}
		}

	}
}

void FractalNetwork::MLPNN(string inputdataname,double min,double max)
{	// the frequency sample data should be normalized first;
	// "min" and "max" refer to the range of the samples;
	// "layernum" refers the number of layers;
	// "nodenum" refers to the number of nodes in each layer and it doesn't include the bias node.
	// Example:
	// layernum=3: 3 layers, one input layer, one output layer and one hidden layer;
	// nodenum=[1,3,1], one node in the input layer, 3 nodes in the hidden layer and one node in the output layer.
	// bias=[1,1], the initial values for bias are 1 in each layer except the output layer;
	int layernum = 6;
	int nodenum[6] = { 1,4,4,5,6,1 };

	Neuron** Layer; // the neuron network, composed of layernum layers and in each layer, there are nodenum[i] nodes.
	double *bias; bias = new double[layernum];
	Layer = new Neuron*[layernum]; Neuron currentnode;
	for (int i = 0; i < layernum; i++) { Layer[i] = new Neuron[nodenum[i]]; }

	// initialization of weights and bias;
	initializeLayer(Layer, layernum, nodenum);
	// assign the input value;
	double *X; double* Y; int capacity = 20000;
	X = new double[capacity]; Y = new double[capacity];
	int DataNum;
	DataNum = inputFittingData(inputdataname, X, Y);
	double truth[1];
	double learningrate = 0.5;
	int activationnum =1;
	double RMS = 10;
	double previouserror = 0;
	double delta = RMS - previouserror;
	int time = 0;
	double criterion = 1.18;// this criterion is adjustable
	while (!(RMS < criterion /DataNum/1.0 && delta<1e-5) )// if error is small and doesn't change, stop
	{
		previouserror = RMS;
		RMS = 0;
		time++;
		for (int dataiter = 0; dataiter <= DataNum; dataiter++)
		{
		
			// assign input value for the input layer, here is just one value;
			Layer[0][0].inputvalue = X[dataiter];
			Layer[0][0].outputvalue = X[dataiter];
			// forward propagation, we need to calculate the output of each neuron based on the weight;
			truth[0] = Y[dataiter];// true data
			ForwordPropagation(Layer, layernum, nodenum, activationnum);// complete the neuron network;
			RMS += abs((truth[0] - Layer[layernum - 1][0].outputvalue))/DataNum;// calculate the error
			BackPropagation(Layer, layernum, nodenum, truth, activationnum,learningrate);// calculate the delta value for each node;
			update_weigth_bias(Layer,  layernum, nodenum,  activationnum,  learningrate);
			
		}
		delta = previouserror - RMS;
	
		if (delta<1e-9 && RMS>criterion / DataNum/1.0)// if the error doesn't decrease any more, but the error is large, we should change the initial values;
		{
			cout << "switch" << "		" << RMS <<"	"<< delta <<endl;
			time = 0;
			// initialization of weights and bias;
			initializeLayer(Layer, layernum, nodenum);
			
		}
	
	}
	cout << RMS <<"		"<<time<< endl;
	system("pause");
	// prediction
	int outputdata = 800; 
	// the number of data points we want to predict;
	// This number decides the resolution of CDF later;
	for (int i = 0; i < outputdata; i++)
	{
		X[i] = min + i * (max-min) / outputdata;
		Layer[0][0].inputvalue = X[i];
		Layer[0][0].outputvalue = X[i];
		ForwordPropagation(Layer, layernum, nodenum, activationnum);
		Y[i] = Layer[layernum - 1][0].outputvalue;

	}
	OutputData("PredictedData.txt", outputdata, X, Y);
	delete[] X; delete[] Y; delete[] bias; 
	for (int i = 0; i < layernum; i++) { delete[]Layer[i];}
	delete[] Layer;
}


double FractalNetwork::area(int start, int end, double *X, double *Y)
{
	double area = 0;
	if (end == start)
	{
		return 0;
	}
	for (int i = start; i < end - 1; i++)
	{
		// area is the area of the trapezoid shape.

		double dx = X[i + 1] - X[i];
		double y1 = Y[i + 1];
		double y2 = Y[i];
		area += (y1 + y2)*dx / 2;
	}
	return area;
}

void FractalNetwork::Generate_fitting_variable(double min, double max, string filename)
{
	double *X; double* Y; int capacity = 200000; double *cdf;
	X = new double[capacity]; Y = new double[capacity]; cdf = new double[capacity];
	int DataNum;
	DataNum = inputFittingData("PredictedData.txt", X, Y);
	double totalarea = area(0, DataNum, X, Y);
	for (int i = 0; i < DataNum; i++)
	{
		cdf[i] = area(0, i,X,Y) / totalarea;
	}
	Outputarray("Cumulative_Distribution", DataNum,cdf);
	int num = 100000;
	// generate 100,000 number of random variables following the corresponding distribution with the inverse CDF method.
	for (int i = 0; i < num; i++)
	{
		double random = randomReal(0, 1, 1, 1);
		int index = FindIndex(cdf, random, 0, DataNum-1); 
		Y[i] = X[index];
	}
	OutputData(filename, num, X, Y);

	delete[] X; delete[] Y; delete[] cdf;
 }



//******************************************************************************************
int FractalNetwork::FindIndex(double *vec, double x, int start, int end) {
	int middle = start + (end - start) / 2;
	if ((end - start) == 1)
		return start;
	else if (x == vec[middle])
	{
		return middle;
	}
	else if (x > vec[middle])
	{
		return FindIndex(vec, x, middle, end);
	}
	else if (x < vec[middle])
	{
		return FindIndex(vec, x, start, middle);
	}

}
//*********************************************************************************************



