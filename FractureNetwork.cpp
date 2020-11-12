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

//*********************************************************************************************
//*********************************************************************************************
//***************************************************************
void FractalNetwork::FractureGrowth(double exponent, int maxlength, int niteration)
{

 
	// seperate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	 a = exponent;
	 lmax = maxlength;
	 a = 1.5;// set a is 2 to make the first generation of fractures large and span the system
	 id = 1;// this is for Ibex cluster 
	 FractureNum = 0;
	 countlongIndex = 0;
	 countS = 0; countL = 0;
	 totalintersection = 0;
	 int *result; result = new int[10];
	 int aa = 0;
	 int bb = 0;
	 int cc = 0;
	 int dd = 0;
	 int rowNum = L / bs;
	 double c = pow(lmax, 1 - a) - pow(lmin, 1 - a);
	 int countintersect = 0;
	 int Nblock = 0;
	 bbn = GenerateBoundary(L);
	 niter = niteration;
	 for (int i = 0; i < rowNum*rowNum; ++i)
		 k[i] = 0;
	 for (int i = 0; i < capacity; i++)
		 m[i] = 2;

	 //**********************************************************************************
	 int n; 
	 n = Generatexycord(L, bs, xcord, ycord);

	 //generate the first fracture;
	 //InputFracture();

	
	  Fracture = GenerateFracture(L, c, FractureNum,1,1,1);
     //Fracture = Fractures[FractureNum];
 
	 if (Length[0]<L)
	 {
		 BlockIndex( Allblocknumber,FractureNum,Nblock,n);

		 tlength = LengthCal(Clipline(Fracture, bbn));
		 LengthSmall[countS] = tlength;
		 AngleSmall[countS] = Angle[FractureNum];
		 countS++;
		
	 }
	 else
	 {
		 longIndex[countlongIndex] = FractureNum;
		 countlongIndex++;

		 tlength = LengthCal(Clipline(Fracture, bbn));
		 LengthLarge[countL] = tlength;
		 AngleLarge[countL] = Angle[FractureNum];
		 countL++;
	 }
	
	 PTR[FractureNum] = -1;
	 FractureLabel[FractureNum] = FractureNum;
	 AllFractureIntersect[FractureNum][0].xcor = Fracture.xpo1;
	 AllFractureIntersect[FractureNum][0].ycor = Fracture.ypo1;
	 AllFractureIntersect[FractureNum][1].xcor = Fracture.xpo2;
	 AllFractureIntersect[FractureNum][1].ycor = Fracture.ypo2;
	 IntersectionWithBoundary(bbn, FractureNum,aa,bb,cc,dd, countintersect, AllFractureIntersect);
	 sort(AllFractureIntersect[FractureNum], AllFractureIntersect[FractureNum] + m[FractureNum], myobject);// sort the intersecting points for forming a line segement;
	 //generate the fracture network

	 while (true)
	 {
		 FractureNum++;
		 countintersect = 0;
		 Nblock = 0;
		 Fracture = GenerateFracture(L,   c, FractureNum,1,1,1);
		// Fracture = Fractures[FractureNum];
		 //cout << FractureNum << endl;
		 PTR[FractureNum] = -1;
		 FractureLabel[FractureNum] = FractureNum;
		 AllFractureIntersect[FractureNum][0].xcor = Fracture.xpo1;
		 AllFractureIntersect[FractureNum][0].ycor = Fracture.ypo1;
		 AllFractureIntersect[FractureNum][1].xcor = Fracture.xpo2;
		 AllFractureIntersect[FractureNum][1].ycor = Fracture.ypo2;
		
		 if (Length[FractureNum ] <= L)
		 {
			 BlockIndex(Allblocknumber, FractureNum, Nblock,n);
			 ZiffClustetCheckNewshort(Allblocknumber, FractureNum,Nblock,countintersect);
		

			 tlength = LengthCal(Clipline(Fracture, bbn));
			 LengthSmall[countS] = tlength;
			 AngleSmall[countS] = Angle[FractureNum];
			 countS++;
		 }
		 else
		 {
			 ZiffClustetCheckNewlong( FractureNum);
			 longIndex[countlongIndex] = FractureNum;
			 countlongIndex++;
			 
			 tlength = LengthCal(Clipline(Fracture, bbn));
			 LengthLarge[countL] = tlength;
			 AngleLarge[countL] = Angle[FractureNum];
			 countL++;
		 }

		 IntersectionWithBoundary(bbn, FractureNum, aa, bb, cc, dd, countintersect, AllFractureIntersect);
	
		 spanindex = BoundaryIntersectionNew( result, aa, bb, cc, dd);
		 if (spanindex.xcor != -NN)
			 break;
	
		 
	 }
	 cout << FractureNum << endl;
	 int spannum = FractureNum + 1;
	 // to generate nucleis to grow
	 bool **Isarrest; Isarrest = new bool*[1000]; // record if the fracture has been arrested.
	 int *Blockrecord; Blockrecord = new int[1000];
	 for (int i = 0; i < 1000; i++) { Isarrest[i] = new bool[2]; Blockrecord[i] = 0; }
	 for (int i = 0; i < 1000; i++) { Isarrest[i][0] = 0; Isarrest[i][1] = 0; }
	 int NucleiNum = 10;// generate 1000 propergated fractures
	 int timestep = 1;
	 pointcor tempPositon;
	 linesegment fracture;
	 double tempAngle, tempLength;
	 pointcor dpositon;
	 for (int i = 0; i < NucleiNum; i++)// Generate new fractures
	 {
		
		 FractureNum++;// fracture number record.
		 tempPositon.xcor = randomReal(0, 1, id, a)*L;
		 tempPositon.ycor = randomReal(0, 1, id, a)*L;
		 tempAngle = (randomReal(0, 1, id, a) * 2 * PI);
		 tempLength = timestep*lmin;
		 dpositon = pol2cart(tempAngle, 0.5*tempLength);
		 fracture.xpo1 = tempPositon.xcor - dpositon.xcor;
		 fracture.ypo1 = tempPositon.ycor - dpositon.ycor;
		 fracture.xpo2 = tempPositon.xcor + dpositon.xcor;
		 fracture.ypo2 = tempPositon.ycor + dpositon.ycor;
		 Fractures[FractureNum] = fracture;
		 Length[FractureNum] = tempLength;
		 Angle[FractureNum] = tempAngle;
		 Position[FractureNum] = tempPositon;
		
		
	 }
	 cout << "good" << FractureNum << endl;
	 system("pause");
	 // check whether all fractures are arrested
	 bool start; bool end;// record the status of start and end point of this fracture;
	 int countarrest = 0;
	 while (true)
	 {
		 timestep++;
		 for (int i = 0; i < NucleiNum; i++)// 
		 {
			 if (Isarrest[i][0] == 0 || Isarrest[i][1] == 0)
			 {
			 
				 Nblock = 0;
				 countintersect = 0;
				 Fracture = Fractures[spannum + i];
				 BlockIndex(Allblocknumber, spannum + i, Nblock, n);
				 if (NucleiCheck(Allblocknumber, spannum + i, Nblock, countintersect, Isarrest[i][0], Isarrest[i][1]))
				 {
					 countarrest++;
				 }
				 else // fracture grows
				 {
					
					 tempLength = timestep*lmin;
					 dpositon = pol2cart(Angle[spannum + i], 0.5*tempLength);
					 if (Isarrest[i][0])// if the start point are cut, then only make the end point grow.
					 {
						
						 Fractures[spannum+i].xpo2 = Position[spannum + i].xcor + dpositon.xcor;
						 Fractures[spannum + i].ypo2 = Position[spannum + i].ycor + dpositon.ycor;
						 Length[spannum + i] = LengthCal(Fractures[spannum + i]);
					 }
					 else if (Isarrest[i][1])
					 {

						 Fractures[spannum + i].xpo1 = Position[spannum + i].xcor - dpositon.xcor;
						 Fractures[spannum + i].ypo1 = Position[spannum + i].ycor - dpositon.ycor;
						 Length[spannum + i] = LengthCal(Fractures[spannum + i]);
					 }
					 else
					 {
						 Fractures[spannum + i].xpo2 = Position[spannum + i].xcor + dpositon.xcor;
						 Fractures[spannum + i].ypo2 = Position[spannum + i].ycor + dpositon.ycor;
						 Fractures[spannum + i].xpo1 = Position[spannum + i].xcor - dpositon.xcor;
						 Fractures[spannum + i].ypo1 = Position[spannum + i].ycor - dpositon.ycor;
						 Length[spannum + i] = LengthCal(Fractures[spannum + i]);
					 }

				 }
			 }
		 }

		 ofstream output2;
		 if (output2.fail())
		 {
			 cout << "openning output 1 file failed" << endl;
			 system("pause");
			 exit(1);
		 }
		 stringstream  filename1;

		 filename1 << "fracture" << timestep << ".txt";
		 output2.open(filename1.str());
		 output2.setf(ios::showpoint);
		 output2.precision(8);
		 for (int i = 0; i < FractureNum + 1; i++)
		 {


			 output2 << Fractures[i].xpo1 << "  " << Fractures[i].ypo1 << "  " << Fractures[i].xpo2 << "  " << Fractures[i].ypo2 << endl;


		 }



		 cout << countarrest << endl;
		 if (countarrest == NucleiNum)
			 break;
	}
	 cout << timestep << endl;







































//	 OutputFracture();
	// calculate the clipped fractures
	 /*
	ClipFractures = new linesegment[capacity];
	for (int i = 0; i<FractureNum + 1; i++)
	{
		pointcor p1, p2;
		p1.xcor = Fractures[i].xpo1; p1.ycor = Fractures[i].ypo1;
		p2.xcor = Fractures[i].xpo2; p2.ycor = Fractures[i].ypo2;
		ClipFractures[i] = Clipline(Fractures[i], bbn);
		 
	}
	OutputClipfractures();
	*/



 OutputPTR();
	//////////////////////////////////////////////////









































 //cout << "Fracture nuk is " << FractureNum << endl;
	 /*
	int countnum = GetBackbone();
	int num = RemovepointinBackbone(countnum);
	//cout << "countnum is " << countnum << "  " << num << endl;
	OutputBackbone(num, "BackboneFinal.txt", BackboneFinal);
	num = AdjacentMatrix(num, BackboneFinal);
	for (int i = 0; i < num; i++)
	{
		BackboneLength[i] = LengthCal(BackboneFinal[i]);
		//cout << BackboneLength[i] << endl;
	}
		 
	 // add more nodes to investigate the influence of aperture;
	 int Backbonenum = ExtendBackbone(num);

	// OutputBackbone(Backbonenum, "BackboneFinal.txt",ExtendedBackbone);

	 FlowForReducedBackbone(Backbonenum, ExtendedBackbone);
	cout << "FractureNum is " << FractureNum << " 1st backbone is " << num << " extended backbone is " << Backbonenum << endl;
	*/

	
	 




	 
	 //***************************************************************************************************************
	

	
}


void FractalNetwork::Damagezone(double exponent, int maxlength, int niteration)
{


	// separate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	innercount = outercount = 0; splaycount = 0; linkcount = 0;
	a = exponent;
	FractureNum = 0;
	//input the fault map
	int FaultNum = InputFracture();
	pointcor center;
	double angle;
	linesegment templine;
	double dx, dy;
	double length;
	int innernum, outernum, splaynum;
	int intensity = 1;// p10 = intensity /m for the most intensive area;
	double scalebar = 15;// unit in m.
	double FaultLength = 0;
	double DamageLength = 0;
	//ParabolaEqu innerboundary1,innerboundary2, outerboundary1,outerboundary2;
	linesegment innerfracture,outerfracture,splayfracture; pointcor intersection1, intersection2;
	cout << FaultNum << endl;
	// generate the damage zone
	for (int i = 0; i < FaultNum; i++)// for each fault segment;
	{
		templine = Fractures[i];
		length = LengthCal(templine);
		FaultLength += length;
		innernum = intensity / (normalCDF(1.0 / length / scalebar,0, 0.2*length) - normalCDF(-1.0 / length / scalebar,0, 0.2*length));// set the most middle 1 m with the intensity.
		cout << innernum << endl;
		system("pause");
		//innernum = 50 * length;
		outernum = 0.2*innernum;
		splaynum = 0.01*outernum;

		// transform the fault segment to the origin point
		center = FindCenter(templine);
		angle = -CalAngle(templine);
		rotate2D(templine, angle, center);
		dx = -center.xcor; dy = -center.ycor;
		Translate2D(templine, dx, dy);

	
		innerboundary1 = CalParabolaVertex(0.5*length, 0.01*length*0.2);
		outerboundary1 = CalParabolaVertex(0.5*length, 0.01*length*0.8);
		innerboundary2 = CalParabolaVertex(0.5*length, -0.01*length*0.2);
		outerboundary2 = CalParabolaVertex(0.5*length, -0.01*length*0.8);

		for (int ii = 0; ii < innernum; ii++)// generate inner damage zone;
		{

			innerfracture = GenerateinnerFracture(length);
			// check intersection of the boundary;
			if (Intersectparabola(innerfracture,innerboundary1, intersection1, intersection2))
			{
				innerfracture.xpo2 = intersection1.xcor;
				innerfracture.ypo2 = intersection1.ycor;
			}
			else if(Intersectparabola(innerfracture, innerboundary2, intersection1, intersection2))
			{
				innerfracture.xpo2 = intersection1.xcor;
				innerfracture.ypo2 = intersection1.ycor;
			}
			// transform the fracture to the original place
			center.xcor = 0; center.ycor = 0;
			rotate2D(innerfracture, -angle, center);
			Translate2D(innerfracture, -dx, -dy);
			innerFractures[innercount] = innerfracture;
			DamageLength += LengthCal(innerfracture);
			innercount++;

		}

		for (int jj = 0; jj < outernum; jj++)// generate outer damage zone;
		{
			outerfracture = GenerateouterFracture(length);
			if (Intersectparabola(outerfracture, outerboundary1, intersection1, intersection2))
			{
				outerfracture.xpo2 = intersection1.xcor;
				outerfracture.ypo2 = intersection1.ycor;
			}
			else if (Intersectparabola(outerfracture, outerboundary2, intersection1, intersection2))
			{
				outerfracture.xpo2 = intersection1.xcor;
				outerfracture.ypo2 = intersection1.ycor;
			}
			else if(Intersectparabola(outerfracture, innerboundary1, intersection1, intersection2))
			{
				outerfracture.xpo2 = intersection1.xcor;
				outerfracture.ypo2 = intersection1.ycor;
			}
			else if (Intersectparabola(outerfracture, innerboundary2, intersection1, intersection2))
			{
				outerfracture.xpo2 = intersection1.xcor;
				outerfracture.ypo2 = intersection1.ycor;
			}
			center.xcor = 0; center.ycor = 0;
			rotate2D(outerfracture, -angle, center);
			Translate2D(outerfracture, -dx, -dy);
			outerFractures[outercount] = outerfracture;
			DamageLength += LengthCal(outerfracture);
			outercount++;
		}
		// generate splay fractures at tips
		/*
		for (int kk = 0; kk < splaynum; kk++)// generate outer damage zone;
		{
			splayfracture = GenerateSplayFracture (length);
			center.xcor = 0; center.ycor = 0;
			rotate2D(splayfracture, -angle, center);
			Translate2D(splayfracture, -dx, -dy);
			splayFractures[splaycount] = splayfracture;
			splaycount++;
		}
		Angle[i] = angle;
		Length[i] = length;
		pointcor translation;
		translation.xcor = dx; translation.ycor = dy;
		Position[i] = translation;
		// connect two fault segments 
		for (int iii = 0; iii < i; iii++)
		{
			// check two segments are in the same sets, which means orientations are almost same
			if (abs(Angle[i] - Angle[iii]) < PI / 18)
			{
				GeneratelinkFracture(i, iii);
			}
		}
		*/
	}
	DamageLength += FaultLength;
	cout << "Fault length is " << FaultLength << "      Damage zone length is " << DamageLength << "		the ratio is " << DamageLength / FaultLength << endl;
	cout << "tipto tip is " << linkcount << "		" << length << endl;
	//system("pause");
	OutputFracture("InnerFractures.txt", innercount,0);
	OutputFracture("OuterFractures.txt", outercount, 1);
	OutputFracture("SplayFractures.txt", splaycount, 2);
	OutputFracture("LinkFractures.txt", linkcount, 3);
}

void FractalNetwork::Damagezone_2(double exponent, int maxlength, int niteration)
{
	// this version is to generate fractures cutting the boundary;
	 
	// separate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	innercount = outercount = 0; splaycount = 0; linkcount = 0;
	a = exponent;
	FractureNum = 0;
	//input the fault map
	int FaultNum = InputFracture();
	pointcor center;
	double angle;
	linesegment templine;
	double dx, dy;
	double length;
	int innernum, outernum, splaynum;
	int intensity = 1;// p10 = intensity /m for the most intensive area;
	double scalebar = 1;// unit in m.
	double FaultLength = 0;
	double DamageLength = 0;
	double innerDamagelength = 0;
	int innercountnew = 0;
	double *P21_inner; P21_inner = new double[20000];
	double P21 = 100;
	//ParabolaEqu innerboundary1,innerboundary2, outerboundary1,outerboundary2;
	linesegment innerfracture, outerfracture, splayfracture; pointcor intersection1, intersection2;
	cout << "faultnum is "<<FaultNum << endl;
	// generate the damage zone
	for (int i = 0; i < FaultNum; i++)// for each fault segment;
	{
		templine = Fractures[i];
		length = LengthCal(templine);
	
		FaultLength += length;
		// the number of fractures in the inner damage zone based on the densiest p10 in the middle of the fracture network.
		innernum = intensity / (normalCDF(1.0 / length / scalebar, 0, 0.2*length) - normalCDF(-1.0 / length / scalebar, 0, 0.2*length));// set the most middle 1 m with the intensity

		// transform the fault segment to the origin point
		////////////////////////////
		center = FindCenter(templine);
		angle = -CalAngle(templine);
 		rotate2D(templine, angle, center);
		dx = -center.xcor; dy = -center.ycor;
		Translate2D(templine, dx, dy);
		////////////////////////////
		// generate boundary for the inner and outer damage zones
		innerboundary1 = CalParabolaVertex(0.5*length, 0.01*length*0.2);
		outerboundary1 = CalParabolaVertex(0.5*length, 0.01*length*0.8);
		innerboundary2 = CalParabolaVertex(0.5*length, -0.01*length*0.2);
		outerboundary2 = CalParabolaVertex(0.5*length, -0.01*length*0.8);
		////////////////////////////////
		double area = 2 * (innerboundary1.A*pow(length, 3) / 12.0 + innerboundary1.C*length);
		// upper limit of the number of fractures in the inner damage zone, we can reduce the number by half and half
		innernum = P21*2.0 *( (innerboundary1.A*pow(length , 3.0) + 12 * innerboundary1.C*length) / 12 - innerboundary1.C)/ abs( innerboundary1.A * pow(0.2*length, 2.0));
		cout << innernum << endl;
	 	 
		// lower limit of the number of fractures in the inner damage zone
	
		double lowerinnernum = P21 * 2 * (innerboundary1.A*pow(length, 3.0) + 12 * innerboundary1.C*length) / 6.0 / length;
	
		innercount = 0;
		innerDamagelength = 0;
		for (int ii = 0; ii < innernum; ii++)// generate inner damage zone;
		{

			innerfracture = GenerateinnerFracture_2(length);// inner fractures are contrained by the boundary.

			// transform the fracture to the original place
			center.xcor = 0; center.ycor = 0;
			rotate2D(innerfracture, -angle, center);
			Translate2D(innerfracture, -dx, -dy);
			innerFractures[innercountnew+innercount] = innerfracture;
		 	innerDamagelength+= LengthCal(innerfracture);
			P21_inner[innercount] = innerDamagelength / area;
			innercount++;

		}
		// try to find the proper number of fractures in the inner damage zone.
		int index = FindIndex(P21_inner,40, 0,innercount- 1);
		cout << "inner index  " << index << endl;
		innercountnew += index;
 		outernum = 0.1*index;
		splaynum = 0.1*outernum;
		for (int jj = 0; jj < outernum; jj++)// generate outer damage zone;
		{
			if (randomReal(0, 1, id, a) > 0.5)
			{
				outerfracture = GenerateouterFracture_2(length);
			}
			else
			{
				outerfracture = GenerateouterFracture(length);
				if (Intersectparabola(outerfracture, outerboundary1, intersection1, intersection2))
				{
					outerfracture.xpo2 = intersection1.xcor;
					outerfracture.ypo2 = intersection1.ycor;
				}
				else if (Intersectparabola(outerfracture, outerboundary2, intersection1, intersection2))
				{
					outerfracture.xpo2 = intersection1.xcor;
					outerfracture.ypo2 = intersection1.ycor;
				}
				else if (Intersectparabola(outerfracture, innerboundary1, intersection1, intersection2))
				{
					outerfracture.xpo2 = intersection1.xcor;
					outerfracture.ypo2 = intersection1.ycor;
				}
				else if (Intersectparabola(outerfracture, innerboundary2, intersection1, intersection2))
				{
					outerfracture.xpo2 = intersection1.xcor;
					outerfracture.ypo2 = intersection1.ycor;
				}
			 
			}
			
			
			center.xcor = 0; center.ycor = 0;
			rotate2D(outerfracture, -angle, center);
			Translate2D(outerfracture, -dx, -dy);
			outerFractures[outercount] = outerfracture;
 			outercount++;
		}
		// generate splay fractures at tips
		Angle[i] = angle;
		Length[i] = length;
		pointcor translation;
		translation.xcor = dx; translation.ycor = dy;
		Position[i] = translation;


		for (int kk = 0; kk < splaynum; kk++)// generate outer damage zone;
		{
		splayfracture = GenerateSplayFracture(length,Fractures[i]);
		splayFractures[splaycount] = splayfracture;
		splaycount++;
		}
		
		
		
		// connect two fault segments
		for (int iii = 0; iii < i; iii++)
		{
		// check two segments are in the same sets, which means orientations are almost same
			double parallellimit = PI / 8;
		if (abs(Angle[i] - Angle[iii]) < parallellimit)
		{
		GeneratelinkFracture(i, iii);
		}
		}
		
	}
	DamageLength += FaultLength;
	cout << "Fault length is " << FaultLength << "      Damage zone length is " << DamageLength << "		the ratio is " << DamageLength / FaultLength << endl;
	cout << "tipto tip is " << linkcount << "		" << length << endl;
	cout << innercountnew << "		" << outercount << "		" << endl;
	//system("pause");
	OutputFracture("InnerFractures.txt", innercountnew, 0);
	OutputFracture("OuterFractures.txt", outercount, 1);
	OutputFracture("SplayFractures.txt", splaycount, 2);
	OutputFracture("LinkFractures.txt", linkcount, 3);
}


void FractalNetwork::Generate2DNetwork(double exponent, int maxlength, int niteration)
{


	// seperate!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	a = exponent;
	lmax = maxlength;
	//a = 1.5;// set a is 2 to make the first generation of fractures large and span the system
	id = 1;// this is for Ibex cluster 
	FractureNum = 0;
	countlongIndex = 0;
	countS = 0; countL = 0;
	totalintersection = 0;
	int *result; result = new int[10];
	int aa = 0;
	int bb = 0;
	int cc = 0;
	int dd = 0;
	int rowNum = L / bs;
	double c = pow(lmax, 1 - a) - pow(lmin, 1 - a);
	int countintersect = 0;
	int Nblock = 0;
	bbn = GenerateBoundary(L);
	niter = niteration;
	for (int i = 0; i < rowNum*rowNum; ++i)
		k[i] = 0;
	for (int i = 0; i < capacity; i++)
		m[i] = 2;

	//**********************************************************************************
	int n;
	n = Generatexycord(L, bs, xcord, ycord);

	//generate the first fracture;
	//InputFracture();


	Fracture = GenerateFracture(L, c, FractureNum, 1, 1, 1);
	//Fracture = Fractures[FractureNum];

	if (Length[0]<L)
	{
		BlockIndex(Allblocknumber, FractureNum, Nblock, n);

		tlength = LengthCal(Clipline(Fracture, bbn));
		LengthSmall[countS] = tlength;
		AngleSmall[countS] = Angle[FractureNum];
		countS++;

	}
	else
	{
		longIndex[countlongIndex] = FractureNum;
		countlongIndex++;

		tlength = LengthCal(Clipline(Fracture, bbn));
		LengthLarge[countL] = tlength;
		AngleLarge[countL] = Angle[FractureNum];
		countL++;
	}

	PTR[FractureNum] = -1;
	FractureLabel[FractureNum] = FractureNum;
	AllFractureIntersect[FractureNum][0].xcor = Fracture.xpo1;
	AllFractureIntersect[FractureNum][0].ycor = Fracture.ypo1;
	AllFractureIntersect[FractureNum][1].xcor = Fracture.xpo2;
	AllFractureIntersect[FractureNum][1].ycor = Fracture.ypo2;
	IntersectionWithBoundary(bbn, FractureNum, aa, bb, cc, dd, countintersect, AllFractureIntersect);
	sort(AllFractureIntersect[FractureNum], AllFractureIntersect[FractureNum] + m[FractureNum], myobject);// sort the intersecting points for forming a line segement;
																										  //generate the fracture network

	while (true)
	{
		FractureNum++;
		countintersect = 0;
		Nblock = 0;
		Fracture = GenerateFracture(L, c, FractureNum, 1, 1, 1);
		// Fracture = Fractures[FractureNum];
		//cout << FractureNum << endl;
		PTR[FractureNum] = -1;
		FractureLabel[FractureNum] = FractureNum;
		AllFractureIntersect[FractureNum][0].xcor = Fracture.xpo1;
		AllFractureIntersect[FractureNum][0].ycor = Fracture.ypo1;
		AllFractureIntersect[FractureNum][1].xcor = Fracture.xpo2;
		AllFractureIntersect[FractureNum][1].ycor = Fracture.ypo2;

		if (Length[FractureNum] <= L)
		{
			BlockIndex(Allblocknumber, FractureNum, Nblock, n);
			ZiffClustetCheckNewshort(Allblocknumber, FractureNum, Nblock, countintersect);


			tlength = LengthCal(Clipline(Fracture, bbn));
			LengthSmall[countS] = tlength;
			AngleSmall[countS] = Angle[FractureNum];
			countS++;
		}
		else
		{
			ZiffClustetCheckNewlong(FractureNum);
			longIndex[countlongIndex] = FractureNum;
			countlongIndex++;

			tlength = LengthCal(Clipline(Fracture, bbn));
			LengthLarge[countL] = tlength;
			AngleLarge[countL] = Angle[FractureNum];
			countL++;
		}

		IntersectionWithBoundary(bbn, FractureNum, aa, bb, cc, dd, countintersect, AllFractureIntersect);

		spanindex = BoundaryIntersectionNew(result, aa, bb, cc, dd);
		if (spanindex.xcor != -NN)
			break;


	}
	cout <<"fracture nu is "<<FractureNum << endl;
 

	OutputFracture("2Dfractures.txt",FractureNum+1);
	 
	OutputPTR();
	 







 
}



void FractalNetwork::GenerateHierarchicalFractures(double exponent, int maxlength, int niteration)
{ /*
	FractureNum = 0;
	niter = niteration;
	Generate_hierarchical_Fracture(L, FractureNum, 0);
	cout << "0 is good " << endl;
	Generate_hierarchical_Fracture(L, FractureNum, 1);
	cout << "1 is good " << endl;
	Generate_hierarchical_Fracture(L, FractureNum, 2);
	for (int i = 0; i < 10; i++)
	{
		Generate_hierarchical_Fracture(L, FractureNum, 1);
		cout << "1 is good " << endl;
		Generate_hierarchical_Fracture(L, FractureNum, 1);
	}
	Output_hierarchical_Fracture(2);
	system("pause");
	*/
	a = exponent;
	lmax = maxlength;
	id = 1;// this is for Ibex cluster 
	FractureNum = -1;
	countlongIndex = 0;
	countS = 0; countL = 0;
	totalintersection = 0;
	int *result; result = new int[10];
	int aa = 0;
	int bb = 0;
	int cc = 0;
	int dd = 0;
	int rowNum = L / bs;
	double c = pow(lmax, 1 - a) - pow(lmin, 1 - a);
	int countintersect = 0;
	int Nblock = 0;
	bbn = GenerateBoundary(L);
	niter = niteration;
	for (int i = 0; i < rowNum*rowNum; ++i)
		k[i] = 0;
	for (int i = 0; i < capacity; i++)
		m[i] = 2;

	//**********************************************************************************
	int n;
	n = Generatexycord(L, bs, xcord, ycord);

	//generate the first fracture;
	//InputFracture();
	for (int i = 0; i <1; i++)
	{
		FractureNum++;
		Generate_hierarchical_Fracture(L, FractureNum, 0);
		Generate_hierarchical_Fracture(L, FractureNum, 1);
		Generate_hierarchical_Fracture(L, FractureNum, 1);
		PTR[FractureNum] = -1;
		FractureLabel[FractureNum] = FractureNum;
		AllFractureIntersect[FractureNum][0].xcor = Fracture.xpo1;
		AllFractureIntersect[FractureNum][0].ycor = Fracture.ypo1;
		AllFractureIntersect[FractureNum][1].xcor = Fracture.xpo2;
		AllFractureIntersect[FractureNum][1].ycor = Fracture.ypo2;
		BlockIndex(Allblocknumber, FractureNum, Nblock, n);
		ZiffClustetCheckNewshort(Allblocknumber, FractureNum, Nblock, countintersect);

		IntersectionWithBoundary(bbn, FractureNum, aa, bb, cc, dd, countintersect, AllFractureIntersect);
	
		longIndex[countlongIndex] = FractureNum;
			countlongIndex++;
	}
	cout << Hierarchiacal_count[0] << endl;
	system("pause");
	while (FractureNum<50)
	{
		FractureNum++;
		Fracture = Generate_hierarchical_Fracture(L, FractureNum, "seed");
		countintersect = 0;
		Nblock = 0;
		//Fracture = GenerateFracture(L, c, FractureNum, 1, 1, 1);
		// Fracture = Fractures[FractureNum];
		//cout << FractureNum << endl;
		PTR[FractureNum] = -1;
		FractureLabel[FractureNum] = FractureNum;
		AllFractureIntersect[FractureNum][0].xcor = Fracture.xpo1;
		AllFractureIntersect[FractureNum][0].ycor = Fracture.ypo1;
		AllFractureIntersect[FractureNum][1].xcor = Fracture.xpo2;
		AllFractureIntersect[FractureNum][1].ycor = Fracture.ypo2;

		BlockIndex(Allblocknumber, FractureNum, Nblock, n);
		ZiffClustetCheckNewshort(Allblocknumber, FractureNum, Nblock, countintersect);


		tlength = LengthCal(Clipline(Fracture, bbn));
		LengthSmall[countS] = tlength;
		AngleSmall[countS] = Angle[FractureNum];
		countS++;


		IntersectionWithBoundary(bbn, FractureNum, aa, bb, cc, dd, countintersect, AllFractureIntersect);

		spanindex = BoundaryIntersectionNew(result, aa, bb, cc, dd);
		if (spanindex.xcor != -NN)
			break;


	}

	Output_hierarchical_Fracture(2);
	OutputPTR();
}
 