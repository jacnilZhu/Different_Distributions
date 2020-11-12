#include "stdafx.h"
#include "myrandom.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <algorithm> 

using namespace std;
void initRandomSeed(int id, double a);


void initRandomSeed(int id, double a) {
	static bool initialized = false;
	if (!initialized) {
		srand(int(time(NULL) + id*a));
		initialized = true;
	}
}

double randomReal(double low, double high, int id, double a) {
	initRandomSeed(id, a);
	double d = rand() / (double(RAND_MAX) + 1);
	double s = d*(high - low);
	return low + s;
}



void RandomPerm(int *arr, int n, int id, double a)
{
	initRandomSeed(id, a);
	random_shuffle(arr, arr + n);

}





int randomInt(int id, double a,int max)
{
	if (max <2)
	{
		return 1;
	}
	initRandomSeed(id, a);
	int d = rand() % max + 1;
	return d;

}