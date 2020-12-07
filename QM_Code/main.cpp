//It is the program of the article: Y. Yan, M. Xu, J.S. Smith et al., Multicamera pedestrian detection using logic minimization, Pattern Recognition, https://doi.org/10.1016/j.patcog.2020.107703
//It is authored by Yuyao Yan, Ming Xu, Jeremy S. Smith, Mo Shen and Jin Xi.
//This program is a x64 program and developed by using Microsoft Visual Studio 2017 (v141) and opencv 2.4.13.
//Program of Subsense is from https://bitbucket.org/pierre_luc_st_charles/subsense/src/master/
//Ground truth is from https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/


#include <opencv2/opencv.hpp> 
#include <cv.h>
#include <cvaux.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <conio.h>
#include <vector>
#include <string>
#include <cassert>

#include "cameraModel.h"
#include "camera.h"
#include "BackgroundSubtractorSuBSENSE.h"

using namespace std;
using namespace cv;

#define Density 3       //grid density, unit is pixels on top view
#define Downsample 2
#define Scale 30		//scale from world coordinate to top view
#define Shift 250		//shift from world coordinate to top view
#define Height 2000     //average height of a pedestrian
#define WHratio 0.35
#define GroundTruthLocation "gt_terrace.txt"
#define TopViewLocation "top.png"
#define Camera_0_location "D:\\video\\Terrace\\terrace1-c0.avi"
#define Camera_1_location "D:\\video\\Terrace\\terrace1-c1.avi"
#define Camera_2_location "D:\\video\\Terrace\\terrace1-c2.avi"
#define Camera_3_location "D:\\video\\Terrace\\terrace1-c3.avi"
#define DeepLabForegroundLocation "D:\\video\\Terrace\\"
#define RSSthreshold 8     //distance, unit is pixels on top view 
#define EDistance 500      //unit is mm
#define LikelihoodThreshold 0.40  //heigher for less candidates
#define PeakThreshold 0.5         //heigher for less candidates
#define QMForegroundPixelRatio 0.2   //a subregion whose duty cycle is less than this number will be ignored 
#define QMRegionAreaThreshold 300    //a subregion whose area is less than this number will be ignored 
#define Visualization 1    //1 for Visualization
#define Evaluation 1       //1 for every 25 frame evaluation, 0 for frame-by-frame result
#define ShowTables 1       //1 for show QM tables
#define ShowPhantoms 0     //1 for show removed phantoms with dashed boxes

Scalar colorful[40] = { CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(64, 0, 64), CV_RGB(0, 255, 128), CV_RGB(255, 255, 255),
CV_RGB(255, 128, 128), CV_RGB(255, 255, 128), CV_RGB(128, 0, 64), CV_RGB(128, 0, 255), CV_RGB(128, 255, 255), CV_RGB(0, 128, 255), CV_RGB(255, 0, 128), CV_RGB(255, 128, 255),
CV_RGB(128, 255, 128), CV_RGB(128, 128, 255), CV_RGB(0, 64, 128), CV_RGB(0, 128, 128), CV_RGB(0, 255, 0), CV_RGB(255, 128, 64), CV_RGB(128, 64, 64), CV_RGB(255, 0, 0), CV_RGB(255, 255, 0),
CV_RGB(64, 0, 64), CV_RGB(0, 255, 128), CV_RGB(255, 255, 255) };

//lookup table for Gaussian
double  tableG[2][16] = { { 0.5, 0.5793, 0.6554, 0.7257, 0.7881, 0.8413, 0.8849, 0.9192, 0.9452, 0.9641, 0.9772, 0.9861, 0.9918, 0.9953, 0.9974, 0.9987 },
{ 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, } };

double Gaussian1D(double dis);
void QMmethod(vector<Camera*> camera, Mat* st, int num, vector<int>& result, vector<int>&);
void QMmap(CvRect rect, Mat* pFrame, int i);
int RemoveDuplates(Mat* A, Mat*B, Mat* fmask);
int removeRow(bool a[], bool b[], int num);
void showtable(int totalNum, int num, bool state[], bool** t, vector<Camera*> camera, bool shortT);
void drawDashRect(CvArr* img, int linelength, int dashlength, CvBlob* blob, CvScalar color, int thickness);
float bbOverlap(Rect box1, Rect box2);
int dfs(int i);
int Hungary();

typedef struct Measure
{
	double prob;
	CvRect R1;
	CvRect R2;
	CvRect R3;
	IplImage* cmask[3];
	int state = 0;
	Point p;
};


int countcolumn[400];
const int N = 100;
int visit[N];
int mark[N];
int match[N][N];
int nd, ng;
int ansH = 0;

int main(int argc, const char * argv[])
{
	//initial groundtruth
	ifstream infile;
	infile.open(GroundTruthLocation);
	assert(infile.is_open());

	string input_s;
	getline(infile, input_s);
	getline(infile, input_s);
	getline(infile, input_s);

	int input_int;
	vector<int> gt_n;
	vector<Rect> gt;

	int countHit = 0;
	int countdet = 0;
	int countGT = 0;
	float MODP = 0;
	float NMODP = 0;
	float NMODA = 0;
	float RECALL, PRECISION, TER, FSCORE = 0;
	int Nframe = 0;

	//initial display parameters 
	char ch;
	int fstep = 0;

	if (Visualization != 0)
	{
		namedWindow("Video", cv::WINDOW_NORMAL);
		resizeWindow("Video", 1220, 576);
		moveWindow("Video", 0, 0);
	}

	//initial camera parameters
	//if you want to change the camera number, you only need to modified here
	vector<Camera*> camera(4);

	camera[0] = new Camera(Downsample, Scale, Shift, Height, WHratio, Camera_0_location);
	camera[0]->cam->setExtrinsic(-4.8441913843e+03, 5.5109448682e+02, 4.9667438357e+03, 1.9007833770e+00, 4.9730769727e-01, 1.8415452559e-01);
	camera[0]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[0]->cam->setIntrinsic(20.161920, 5.720865e-04, 366.514507, 305.832552, 1);
	camera[0]->cam->internalInit();

	camera[1] = new Camera(Downsample, Scale, Shift, Height, WHratio, Camera_1_location);
	camera[1]->cam->setExtrinsic(-65.433635, 1594.811988, 2113.640844, 1.9347282363e+00, -7.0418616982e-01, -2.3783238362e-01);
	camera[1]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[1]->cam->setIntrinsic(19.529144, 5.184242e-04, 360.228130, 255.166919, 1);
	camera[1]->cam->internalInit();

	camera[2] = new Camera(Downsample, Scale, Shift, Height, WHratio, Camera_2_location);
	camera[2]->cam->setExtrinsic(1.9782813424e+03, -9.4027627332e+02, 1.2397750058e+04, -1.8289537286e+00, 3.7748154985e-01, 3.0218614321e+00);
	camera[2]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[2]->cam->setIntrinsic(19.903218, 3.511557e-04, 355.506436, 241.205640, 1.0000000000e+00);
	camera[2]->cam->internalInit();

	camera[3] = new Camera(Downsample, Scale, Shift, Height, WHratio, Camera_3_location);
	camera[3]->cam->setExtrinsic(4.6737509054e+03, -2.5743341287e+01, 8.4155952460e+03, -1.8418460467e+00, -4.6728290805e-01, -3.0205552749e+00);
	camera[3]->cam->setGeometry(720, 576, 576, 576, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02, 2.3000000000e-02);
	camera[3]->cam->setIntrinsic(20.047015, 4.347668e-04, 349.154019, 245.786168, 1);
	camera[3]->cam->internalInit();

#pragma omp parallel for
	for (int i = 0; i < camera.size(); i++)
	{
		camera[i]->initialBG(1);
		camera[i]->mapToTop();
	}

	//initial topview maps
	Mat topback = imread(TopViewLocation);
	static Mat map(1000, 1000, CV_32F, Scalar(0));
	static Mat mapTop(1000, 1000, CV_8UC1, Scalar(0));
	Mat topCross(1000, 1000, CV_8UC3);

	//initial parameters to store the intermediate results
	int tempvalue1;
	int tempvalue2;
	int num = 0;
	static Mat st(1, 400, CV_8U, Scalar(0));
	static Measure mea[400];
	int ex = 0;
	vector<int> result;
	vector<int> rstnum;
	double X, Y, Z = 0;
	double x, y;
	Point pl[4];
	Point LT(285, 290), RB(450, 460);
	Point a(285, 290), b(450, 290), c(450, 460), d(285, 460);

	while (1)
	{
		//read next frame
		for (int i = 0; i < camera.size(); i++)
			camera[i]->readNextFrame();

		if (camera[0]->FrameNumber > 5000)
			break;

		if (Evaluation)
		{
			//read groundtruth
			gt_n.clear();
			for (int i_in = 0; i_in < 9; i_in++)
			{
				infile >> input_int;
				if (input_int >= 0)
					gt_n.push_back(input_int);
			}
		}
		if (Evaluation)
		{
			tempvalue1 = 5;
			tempvalue2 = 25;
		}
		else
		{
			tempvalue1 = 1;
			tempvalue2 = 1;
		}

		//update background model every 5 frames or frame number is less than 50
		if (camera[0]->FrameNumber < 50 || (camera[0]->FrameNumber % tempvalue1 == 0 && camera[0]->FrameNumber % tempvalue2 != 0))
		{
#pragma omp parallel for
			for (int i = 0; i < camera.size(); i++)
				camera[i]->updateMog();
		}
		
		if (!Evaluation)
			tempvalue2 = 1;
		//detect pedestrain every 25 frames
		if (camera[0]->FrameNumber >= 50 && camera[0]->FrameNumber % tempvalue2 == 0)
		{
			cout << camera[0]->FrameNumber << endl;
			string fnum = to_string(camera[0]->FrameNumber);
			while (fnum.size() < 4) fnum = "0" + fnum;

#pragma omp parallel for
			for (int i = 0; i < camera.size(); i++)
			{
				camera[i]->foreground.release();
				camera[i]->updateMog();
			}

#pragma omp parallel for
			for (int i = 0; i < camera.size(); i++)
				integral(camera[i]->foreground / 255.0, camera[i]->iiimage, CV_32S);

			//initial top view and draw AOI on each camera view
			int flag;
			int flag1;
			int flag2;
			map.setTo(1);

			if (Visualization > 0)
			{
				topCross.setTo(0);
				topCross += topback;
				rectangle(topCross, Rect(LT, RB), CV_RGB(255, 0, 0), 2);
				rectangle(topCross, Rect(125, 125, 500, 500), CV_RGB(255, 255, 255), 1);

				for (int k = 0; k < camera.size(); k++)
				{
					Z = 0;

					X = (a.x - Shift) * Scale;
					Y = (a.y - Shift) * Scale;
					camera[k]->cam->worldToImage(X, Y, Z, x, y);
					pl[0].x = x / Downsample;
					pl[0].y = y / Downsample;

					X = (b.x - Shift) * Scale;
					Y = (b.y - Shift) * Scale;
					camera[k]->cam->worldToImage(X, Y, Z, x, y);
					pl[1].x = x / Downsample;
					pl[1].y = y / Downsample;

					X = (c.x - Shift) * Scale;
					Y = (c.y - Shift) * Scale;
					camera[k]->cam->worldToImage(X, Y, Z, x, y);
					pl[2].x = x / Downsample;
					pl[2].y = y / Downsample;

					X = (d.x - Shift) * Scale;
					Y = (d.y - Shift) * Scale;
					camera[k]->cam->worldToImage(X, Y, Z, x, y);
					pl[3].x = x / Downsample;
					pl[3].y = y / Downsample;

					line(camera[k]->frame, pl[0], pl[1], CV_RGB(255, 0, 0), 2);
					line(camera[k]->frame, pl[1], pl[2], CV_RGB(255, 0, 0), 2);
					line(camera[k]->frame, pl[2], pl[3], CV_RGB(255, 0, 0), 2);
					line(camera[k]->frame, pl[3], pl[0], CV_RGB(255, 0, 0), 2);
				}
			}

			//calculate duty cycle of locations on top view
			double tempv;
			for (int i = 0; i < 1000; i = i + Density)
				for (int j = 0; j < 1000; j = j + Density)
				{
					circle(topCross, Point(i, j), 0, CV_RGB(50, 50, 50), 1);
					flag = 0;
					flag1 = 0;
					flag2 = 0;

					for (int k = 0; k < camera.size(); k++)
					{
						if (camera[k]->topshow.at<uchar>(i, j) == 1)
							flag1++;
						if (camera[k]->top.at<uchar>(i, j) == 1)
						{
							flag2++;
							flag++;
						}
					}

					if (flag1 >= 2)
					{
						for (int k = 0; k < camera.size(); k++)
						{
							if (camera[k]->topshow.at<uchar>(i, j) == 1)
							{
								tempv = camera[k]->dutyCycle(i, j);
								if (tempv >= LikelihoodThreshold)
									map.at<float>(i, j) *= tempv;
								else
								{
									map.at<float>(i, j) = 0;
									break;
								}
							}
						}
					}
					else if (flag1 >= 1 && flag >= 1)
					{
						if (i >= LT.x && i <= RB.x && j >= LT.y  && j <= RB.y)
						{
							for (int k = 0; k < camera.size(); k++)
							{
								if (camera[k]->topshow.at<uchar>(i, j) == 1)
								{
									tempv = camera[k]->dutyCycle(i, j);
									if (tempv >= LikelihoodThreshold)
										map.at<float>(i, j) *= camera[k]->dutyCycle(i, j);
									else
									{
										map.at<float>(i, j) = 0;
										break;
									}
								}
							}
						}
					}

					if (map.at<float>(i, j) == 1)
						map.at<float>(i, j) = 0;
				}

			//find the peaks and store in 'mea'
			st.setTo(0);
			num = 0;
			int thNum;
			int flagv;
			for (int i = Density; i < 1000 - Density; i = i + Density)
				for (int j = Density; j < 1000 - Density; j = j + Density)
				{
					ex = 0;
					thNum = 0;
					for (int ii = i - Density; ii <= i + Density; ii = ii + Density)
						for (int jj = j - Density; jj <= j + Density; jj = jj + Density)
						{
							if (map.at<float>(i, j) > map.at<float>(ii, jj))
								ex++;
						}

					for (int k = 0; k < camera.size(); k++)
					{
						thNum += camera[k]->topshow.at<uchar>(i, j);
					}

					if ((ex == 8) && (map.at<float>(i, j) > pow(PeakThreshold, thNum)))
						if (i >= LT.x - 100 && i <= RB.x + 100 && j >= LT.y - 100 && j <= RB.y + 100)
						{
							flag = 0;
							flag1 = 0;
							flag2 = 0;

							for (int k = 0; k < camera.size(); k++)
							{
								if (camera[k]->topshow.at<uchar>(i, j) == 1)
									flag1++;
								if (camera[k]->top.at<uchar>(i, j) == 1)
								{
									flag2++;
									if (camera[k]->foreground.at<uchar>(camera[k]->maph.at<Point2f>(i, j).y, camera[k]->maph.at<Point2f>(i, j).x) > 0)
										flag++;
								}

							}

							mea[num].prob = map.at<float>(i, j);
							mea[num].p.x = i;
							mea[num].p.y = j;
							num++;
						}
				}

			//calculate foot and head likelihoods, and RSS filter
			int num2 = 0;
			bool flag_delete = 1;
			double max_temp;
			Point max_point(-1, -1);
			bool flag_count;
			while (flag_delete)
			{
				flag_delete = 0;
				max_temp = 0;
				max_point = Point(-1, -1);

				for (int i = 0; i < num; i++)
				{
					if (mea[i].prob > max_temp)
					{
						max_temp = mea[i].prob;
						max_point.x = mea[i].p.x;
						max_point.y = mea[i].p.y;
					}
				}

				flag_count = 0;
				if (max_point.x != -1)
				{
					for (int k = 0; k < camera.size(); k++)
					{
						camera[k]->p[num2] = max_point;
						if (camera[k]->top.at<uchar>(max_point.x, max_point.y) == 1)
						{
							st.at<uchar>(0, num2) += pow(2, k);
							camera[k]->r[num2] = Rect(camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[0],
								camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[1],
								camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[2],
								camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3]);

							camera[k]->prob[num2] = camera[k]->dutyCycle(max_point.x, max_point.y);

							Mat rst(1, camera[k]->rctMap.at<Vec4f>(max_point.x, max_point.y)[3], CV_64FC1, Scalar(0));
							Mat roim = camera[k]->foreground(camera[k]->r[num2]);
							Mat roidoublem;
							roim.convertTo(roidoublem, CV_64FC1, 1 / 255.0);
							reduce(roidoublem, rst, 1, CV_REDUCE_SUM);

							double B;
							double T;

							//foot likelihood
							flagv = 0;
							int vcounter = 0;
							if (camera[k]->r[num2].width > 0 && ((double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height > 0.2) && (double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height < 0.5)
								for (int m = camera[k]->r[num2].height - 1; m >= 0; m--)
								{
									if (flagv == 0)
									{
										if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
											vcounter++;
										else
											flagv++;
									}
									else
									{
										if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
										{
											vcounter = vcounter + flagv;
											flagv = 0;
										}
										else
											flagv++;
									}
									if (flagv >= 5)
									{
										B = m + 4;
										break;
									}
								}
							else
							{
								B = camera[k]->r[num2].height - 1;
							}
							camera[k]->probB[num2] = Gaussian1D((camera[k]->r[num2].height - (B + 1)) / (camera[k]->r[num2].height * 1 / 10)) * 2;

							//head likelihood
							flagv = 0;
							vcounter = 0;
							if (camera[k]->r[num2].width > 0 && ((double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height > 0.2) && (double)camera[k]->r[num2].width / (double)camera[k]->r[num2].height < 0.5)
								for (int m = 0; m < camera[k]->r[num2].height; m++)
								{
									if (flagv == 0)
									{
										if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
											vcounter++;
										else
											flagv++;
									}
									else
									{
										if (rst.at<double>(0, m) <= camera[k]->r[num2].width / 10.0)
										{
											vcounter = vcounter + flagv;
											flagv = 0;
										}
										else
											flagv++;
									}
									if (flagv >= 5)
									{
										T = m - 4;
										break;
									}
								}
							else
							{
								T = 0;
							}
							camera[k]->probT[num2] = Gaussian1D(T / (camera[k]->r[num2].height * 2 / 5)) * 2;
						}
						else
						{
							camera[k]->prob[num2] = 1;
							camera[k]->probB[num2] = 1;
							camera[k]->probT[num2] = 1;
						}
					}

					num2++;

					//RSS filter
					for (int i = 0; i < num; i++)
					{
						if (mea[i].p.x != -1)
						{
							if (sqrt((mea[i].p.x - max_point.x)*(mea[i].p.x - max_point.x) + (mea[i].p.y - max_point.y)*(mea[i].p.y - max_point.y)) <= RSSthreshold)
							{
								mea[i].prob = -1;
								mea[i].p.x = -1;
								mea[i].p.y = -1;
								flag_delete = 1;
							}
						}
					}

				}
			}

			//draw contours on frame
			if (Visualization > 0)
			{
				vector<vector<Point>> contour;
				for (int k = 0; k < camera.size(); k++)
				{
					findContours(camera[k]->foreground, contour, RETR_LIST, CHAIN_APPROX_NONE);
					drawContours(camera[k]->frame, contour, -1, CV_RGB(0, 255, 0), 1);
				}
			}

			//run QM method
			result.clear();
			rstnum.clear();
			QMmethod(camera, &st, num2, result, rstnum);

			if (Visualization > 0)
			{
				bool flagrst = 0;
				int ii = 1;
				for (int i = 0; i < num2; i++)
				{
					flagrst = 0;
					for (int j = 0; j < rstnum.size(); j++)
					{
						if (i == rstnum[j])
							flagrst = 1;
					}
					if (flagrst == 1)
					{
						circle(topCross, camera[0]->p[i], 4, colorful[ii], -1);
						ii++;
					}
				}
				flip(topCross, topCross, 0);
			}

			//evaluation
			if (Evaluation)
			{
				float overlap;
				float gdIOU = 0;
				float totIOU = 0;
				int tNframe = 0;
				int canum, frHit = 0;
				float tfrIOU = 0;
				float distance;
				int ga, gb, gn;
				float wga, wgb, wdx, wdy;
				int wk = 0;
				memset(match, 0, sizeof(match));

				if (gt_n.size() > 0)
				{
					for (int gt_in = 0; gt_in < gt_n.size(); gt_in++)
					{
						ga = gt_n[gt_in] % 30 * 250 + 125 - 500;
						gb = gt_n[gt_in] / 30 * 250 + 125 - 1500;

						wga = gt_n[gt_in] % 30 * 250 + 125 - 500;
						wgb = gt_n[gt_in] / 30 * 250 + 125 - 1500;

						ga = ga / 30 + 250;
						gb = gb / 30 + 250;

						flip(topCross, topCross, 0);
						circle(topCross, Point(ga, gb), 17, Scalar(0, 0, 255), 1);
						flip(topCross, topCross, 0);

						for (int i = 0; i < result.size(); i++)
						{

							wdx = (camera[0]->p[result[i]].x - 250) * 30;
							wdy = (camera[0]->p[result[i]].y - 250) * 30;

							distance = abs(sqrt((wdx - wga)*(wdx - wga)
								+ (wdy - wgb)*(wdy - wgb)));
							if (distance <= EDistance)
							{
								match[gt_in][i] = 1;
							}
						}
					}
				}
				ng = gt_n.size();
				nd = result.size();
				ansH = Hungary();

				if (gt_n.size() > 0)
				{
					for (int j = 0; j < ng; j++)
					{
						int detIdx = -1;
						for (int i = 0; i < nd; i++)
						{
							if (mark[i] == j)
							{
								detIdx = i;
								break;
							}
						}

						ga = gt_n[j] % 30 * 250 + 125 - 500;
						gb = gt_n[j] / 30 * 250 + 125 - 1500;
						ga = ga / 30 + 250;
						gb = gb / 30 + 250;

						if (detIdx != -1)
						{
							if ((ga >= LT.x - 5 && ga <= RB.x + 5 && gb >= LT.y - 5 && gb <= RB.y + 5) || (camera[0]->p[result[detIdx]].x >= LT.x - 5 && camera[0]->p[result[detIdx]].x <= RB.x + 5 && camera[0]->p[result[detIdx]].y >= LT.y - 5 && camera[0]->p[result[detIdx]].y <= RB.y + 5))
							{
								countGT++;
								countdet++;
								countHit++;
								frHit++;
								tNframe++;

								for (int k = 0; k < camera.size(); k++)
								{
									if (camera[k]->topshow.at<uchar>(ga, gb) == 1)
									{
										overlap = bbOverlap(Rect(camera[k]->rctMap.at<Vec4f>(ga, gb)[0], camera[k]->rctMap.at<Vec4f>(ga, gb)[1],
											camera[k]->rctMap.at<Vec4f>(ga, gb)[2], camera[k]->rctMap.at<Vec4f>(ga, gb)[3]),
											Rect(camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[0], camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[1],
												camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[2], camera[k]->rctMap.at<Vec4f>(int(camera[k]->p[result[detIdx]].x), int(camera[k]->p[result[detIdx]].y))[3]));

										gdIOU += overlap;
										canum += 1;
									}
								}

								tfrIOU = gdIOU / canum;
								totIOU += tfrIOU;

								canum = 0;
								tfrIOU = 0;
								gdIOU = 0;
							}
						}
						else
						{
							if (ga >= LT.x + 5 && ga <= RB.x - 5 && gb >= LT.y + 5 && gb <= RB.y - 5)
							{
								flip(topCross, topCross, 0);
								circle(topCross, Point(ga, gb), 17, Scalar(255, 0, 0), 1);
								flip(topCross, topCross, 0);

								countGT++;
								wk = 1;
							}
						}
					}

					for (int i = 0; i < nd; i++)
					{
						if (mark[i] == -1)
						{
							if (camera[0]->p[result[i]].x >= LT.x + 5 && camera[0]->p[result[i]].x <= RB.x - 5 && camera[0]->p[result[i]].y >= LT.y + 5 && camera[0]->p[result[i]].y <= RB.y - 5)
							{
								countdet++;
								wk = 1;
							}
						}
					}

					if (tNframe > 0)
					{
						Nframe++;
					}

					if (totIOU > 0)
					{
						MODP += totIOU / frHit;
					}
					totIOU = 0;
					frHit = 0;
				}
			}

			if (Visualization != 0)
			{
				Mat comb = Mat(576, 1220, CV_8UC3, Scalar(0));
				Mat ist = comb(Rect(0, 0, 360, 288));
				camera[1]->frame.copyTo(ist);
				ist = comb(Rect(360, 0, 360, 288));
				camera[0]->frame.copyTo(ist);
				ist = comb(Rect(0, 288, 360, 288));
				camera[2]->frame.copyTo(ist);
				ist = comb(Rect(360, 288, 360, 288));
				camera[3]->frame.copyTo(ist);

				ist = comb(Rect(720, 0, 500, 500));
				flip(topCross, topCross, 0);
				Mat topist = topCross(Rect(125, 125, 500, 500));
				flip(topist, topist, 0);
				topist.copyTo(ist);
				flip(topist, topist, 0);
				flip(topCross, topCross, 0);

				imshow("Video", comb); waitKey(1);
			}

		}

		//"space" to pause and "enter" to continue
		if (fstep == 1) {
			ch = getch();
			if (ch == 13) fstep = 0;
			if (ch == 27) break;
		}
		else {
			if (kbhit()) {
				ch = getch();
				if (ch == 32) fstep = 1;
				else break;
			}
		}


	}
	if (Evaluation)
	{
		NMODP = MODP / Nframe;
		TER = ((countGT - countHit) + (countdet - countHit)) / double(countGT);
		PRECISION = countHit / double(countdet);
		RECALL = countHit / double(countGT);
		NMODA = 1 - TER;
		FSCORE = 2 * PRECISION*RECALL / (PRECISION + RECALL);

		cout << endl << endl << endl;
		cout << "count GT: " << countGT << " " << "count detection: " << countdet << " " << "count matched: " << countHit << endl;
		cout << "FN " << countGT - countHit << endl;
		cout << "FN rate" << (countGT - countHit) / double(countGT) << endl;
		cout << "FP " << countdet - countHit << endl;
		cout << "FP rate" << (countdet - countHit) / double(countdet) << endl;
		cout << "MDR " << (countGT - countHit) / double(countGT) << endl;
		cout << "FDR " << (countdet - countHit) / double(countGT) << endl;
		cout << "TER " << ((countGT - countHit) + (countdet - countHit)) / double(countGT) << endl;
		cout << "PRE " << countHit / double(countdet) << endl;
		cout << "REC " << countHit / double(countGT) << endl;
		cout << "F-Score: " << FSCORE << endl;
		cout << "N-MODA: " << NMODA << endl;
		cout << "N-MODP: " << NMODP << endl;
	}

	system("pause");

	return 0;
}

float bbOverlap(Rect box1, Rect box2)
{
	if (box1.x > box2.x + box2.width) { return 0.0; }
	if (box1.y > box2.y + box2.height) { return 0.0; }
	if (box1.x + box1.width < box2.x) { return 0.0; }
	if (box1.y + box1.height < box2.y) { return 0.0; }
	float colInt = min(box1.x + box1.width, box2.x + box2.width) - max(box1.x, box2.x);
	float rowInt = min(box1.y + box1.height, box2.y + box2.height) - max(box1.y, box2.y);
	float intersection = colInt * rowInt;
	float area1 = box1.width*box1.height;
	float area2 = box2.width*box2.height;
	return intersection / (area1 + area2 - intersection);
}

void QMmethod(vector<Camera*> camera, Mat* st, int num, vector<int>& result, vector<int>& rstnum)
{
	int n = 0;
	CvFont font;
	char text[10];
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, 0.5, 0, 2);

	bool state[400];
	double prob[400];
	int camNum[400];

	if (ShowTables)
	{
		cout << endl;
		cout << "\t\b\b\b\b" << "  ";

		for (int k = 0; k < camera.size(); k++)
		{
			cout << k + 1 << "F    " << k + 1 << "T    " << k + 1 << "B    ";
		}
		cout << " JL" << endl;
	}

	for (int i = 0; i < num; i++)
	{
		if (ShowTables)
		{
			cout << "I" << i << "\t\b\b\b\b";
		}
		prob[i] = 1;
		camNum[i] = (int)camera.size();
		for (int k = 0; k < camera.size(); k++)
		{
			if (ShowTables)
			{
				cout << fixed << setw(5) << setprecision(3) << camera[k]->prob[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probT[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probB[i] << " ";
			}
			prob[i] *= camera[k]->prob[i] * camera[k]->probT[i] * camera[k]->probB[i];
			if (camera[k]->prob[i] == 1)
				camNum[i]--;

		}
		prob[i] = pow(prob[i], 1.0 / camNum[i]);
		if (ShowTables)
		{
			cout << " " << prob[i] << endl;
		}
	}

	//delete regions which have low overall likelihood
	int inx = 0;
	while (inx < num)
	{
		if (prob[inx] < PeakThreshold)
		{
			for (int j = inx; j <= num - 1; j++)
			{
				for (int m = 0; m < camera.size(); m++)
				{
					camera[m]->r[j] = camera[m]->r[j + 1];
					camera[m]->prob[j] = camera[m]->prob[j + 1];
					camera[m]->probB[j] = camera[m]->probB[j + 1];
					camera[m]->probT[j] = camera[m]->probT[j + 1];
					camera[m]->p[j] = camera[m]->p[j + 1];
					prob[j] = prob[j + 1];
					st->at<uchar>(0, j) = st->at<uchar>(0, j + 1);
				}
			}
			num--;
			inx--;
		}
		inx++;
	}

	if (ShowTables)
	{
		cout << endl;
		cout << "Modified Table" << endl;
		cout << "\t\b\b\b\b" << "  ";

		for (int k = 0; k < camera.size(); k++)
		{
			cout << k + 1 << "F    " << k + 1 << "T    " << k + 1 << "B    ";
		}
		cout << " JL" << endl;
	}

	for (int i = 0; i < num; i++)
	{
		if (ShowTables)
		{
			cout << "I" << i << "\t\b\b\b\b";
		}
		prob[i] = 1;
		camNum[i] = (int)camera.size();
		for (int k = 0; k < camera.size(); k++)
		{
			if (ShowTables)
			{
				cout << fixed << setw(5) << setprecision(3) << camera[k]->prob[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probT[i] << " " << fixed << setw(5) << setprecision(3) << camera[k]->probB[i] << " ";
			}
			prob[i] *= camera[k]->prob[i] * camera[k]->probT[i] * camera[k]->probB[i];
			if (camera[k]->prob[i] == 1)
				camNum[i]--;

		}
		prob[i] = pow(prob[i], 1.0 / camNum[i]);
		if (ShowTables)
		{
			cout << " " << prob[i] << endl;
		}
	}

	for (int i = 0; i < camera.size(); i++)
	{
		camera[i]->pF.setTo(0);
		camera[i]->T.setTo(0);
	}

	for (int i = 0; i < num; i++)
	{
		state[i] = 0;
		for (int k = 0; k < camera.size(); k++)
			if (st->at<uchar>(0, i) >> k & 0x01)
				QMmap(camera[k]->r[i], &camera[k]->pF, i);
	}

	for (int k = 0; k < camera.size(); k++)
		camera[k]->Tnum = RemoveDuplates(&camera[k]->pF, &camera[k]->T, &camera[k]->foreground);


	bool** t = new bool*[num];

	for (int k = 0; k < camera.size(); k++)
	{
		for (int j = 0; j < camera[k]->Tnum; j++)
		{
			if (camera[k]->T.at<unsigned long long>(j, 0) == 0)
			{
				for (int i = j; i < camera[k]->Tnum - 1; i++)
				{
					camera[k]->T.at<unsigned long long>(i, 0) = camera[k]->T.at<unsigned long long>(i + 1, 0);
					camera[k]->T.at<unsigned long long>(i, 1) = camera[k]->T.at<unsigned long long>(i + 1, 1);
					camera[k]->T.at<unsigned long long>(i, 2) = camera[k]->T.at<unsigned long long>(i + 1, 2);
				}
				break;
			}
		}
	}

	int totalNum = 0;
	for (int k = 0; k < camera.size(); k++)
	{
		camera[k]->Tnum--;
		totalNum += camera[k]->Tnum;
	}

	for (int i = 0; i < num; i++)
		t[i] = new bool[totalNum];

	int idx;
	for (int i = 0; i < num; i++)
	{
		idx = 0;
		for (int k = 0; k < camera.size(); k++)
			for (int j = 0; j < camera[k]->Tnum; j++, idx++)
				t[i][idx] = camera[k]->T.at<unsigned long long>(j, 0) >> i & 0x01;
	}

	if (ShowTables)
	{
		showtable(totalNum, num, state, t, camera, false);

		cout << "*********** Delete Small Sunregion *********" << endl << endl;
	}
	idx = 0;
	for (int k = 0; k < camera.size(); k++)
	{
		for (int i = 0; i < camera[k]->Tnum; i++, idx++)
			if ((double)camera[k]->T.at<unsigned long long>(i, 2) / camera[k]->T.at<unsigned long long>(i, 1) < QMForegroundPixelRatio || (double)camera[k]->T.at<unsigned long long>(i, 2) < QMRegionAreaThreshold)
				for (int j = 0; j < num; j++)
					t[j][idx] = 0;
	}
	if (ShowTables)
	{
		cout << totalNum << endl;
	}
	for (int j = 0; j < totalNum; j++)
	{
		countcolumn[j] = 0;

		for (int i = 0; i < num; i++)
		{
			if (t[i][j] == 1)
			{
				countcolumn[j] = 1;
				continue;
			}
		}
	}
	if (ShowTables)
	{
		showtable(totalNum, num, state, t, camera, true);
	}

	//QM-Method
	int kmarker = 0;
	int imarker = 0;

	int sum;
	int* jmarker = new int[num];

	int tempnum;
	int* iIndex = new int[num];
	for (int i = 0; i < num; i++)
		iIndex[i] = 0;

	bool* ch = new bool[num];
	bool* nch = new bool[num];

	bool loop = 1;
	while (loop)
	{
		loop = 0;
		for (int i = 0; i < num; i++)
		{
			tempnum = 0;
			for (int j = 0; j < totalNum; j++)
			{
				tempnum += t[i][j];
			}
			if (iIndex[i] != tempnum)
			{
				iIndex[i] = tempnum;
				loop = 1;
			}
		}
		if (loop == 0)
			break;
		if (ShowTables)
		{
			cout << "*************** First  Step ****************" << endl;
		}

		for (int j = 0; j < num; j++)
			jmarker[j] = 0;

		for (int i = 0; i < totalNum; i++)
		{
			sum = 0;
			for (int j = 0; j < num; j++)
				sum += t[j][i];
			if (sum == 1)
			{
				for (int j = 0; j < num; j++)
					if (t[j][i] == 1)
						jmarker[j] = 1;
			}
		}

		for (int j = 0; j < num; j++)
		{
			if (jmarker[j] == 1)
			{
				for (int k = 0; k < totalNum; k++)
				{
					if (t[j][k] == 1)
						for (int n = 0; n < num; n++)
							t[n][k] = 0;
				}
				state[j] = 1;
			}
		}

		if (ShowTables)
		{
			showtable(totalNum, num, state, t, camera, true);
		}

		loop = 0;
		for (int i = 0; i < num; i++)
		{
			tempnum = 0;
			for (int j = 0; j < totalNum; j++)
			{
				tempnum += t[i][j];
			}
			if (iIndex[i] != tempnum)
			{
				iIndex[i] = tempnum;
				loop = 1;
			}
		}
		tempnum = 0;
		for (int i = 0; i < num; i++)
		{
			tempnum += iIndex[i];
		}
		if (loop == 0 || tempnum == 0)
			break;

		if (ShowTables)
		{
			cout << "*************** Second  Step ****************" << endl;
		}
		for (int i = 0; i < num; i++)
			for (int j = 0; j < num; j++)
			{
				if (i != j)
				{
					if (removeRow(t[i], t[j], totalNum) == 1)
					{
						for (int k = 0; k < totalNum; k++)
							t[i][k] = 0;
					}
					else if (removeRow(t[i], t[j], totalNum) == 2)
					{
						if (prob[i] < prob[j])
						{
							for (int k = 0; k < totalNum; k++)
								t[i][k] = 0;
						}
					}
				}
			}
		if (ShowTables)
		{
			showtable(totalNum, num, state, t, camera, true);

			cout << "*************** Third  Step ****************" << endl;
		}

		for (int j = 0; j < num; j++)
			jmarker[j] = 0;

		for (int i = 0; i < totalNum; i++)
		{
			sum = 0;
			for (int j = 0; j < num; j++)
				sum += t[j][i];
			if (sum == 1)
			{
				for (int j = 0; j < num; j++)
					if (t[j][i] == 1)
						jmarker[j] = 1;
			}
		}

		for (int j = 0; j < num; j++)
		{
			if (jmarker[j] == 1)
			{
				for (int k = 0; k < totalNum; k++)
				{
					if (t[j][k] == 1)
						for (int n = 0; n < num; n++)
							t[n][k] = 0;
				}
				state[j] = 1;
			}
		}

		if (ShowTables)
		{
			showtable(totalNum, num, state, t, camera, true);

			cout << "*************** Fourth  Step ****************" << endl;
		}

		for (int i = 0; i < num; i++)
			for (int j = 0; j < num; j++)
			{
				if (i != j)
				{
					if (removeRow(t[i], t[j], totalNum) == 1)
					{
						for (int k = 0; k < totalNum; k++)
							t[i][k] = 0;
					}
					else if (removeRow(t[i], t[j], totalNum) == 2)
					{
						if (prob[i] < prob[j])
						{
							for (int k = 0; k < totalNum; k++)
								t[i][k] = 0;
						}
					}
				}
			}
		if (ShowTables)
		{
			showtable(totalNum, num, state, t, camera, true);
		}

		loop = 0;
		for (int i = 0; i < num; i++)
		{
			tempnum = 0;
			for (int j = 0; j < totalNum; j++)
			{
				tempnum += t[i][j];
			}
			if (iIndex[i] != tempnum)
			{
				iIndex[i] = tempnum;
				loop = 1;
			}
		}
		tempnum = 0;
		for (int i = 0; i < num; i++)
		{
			tempnum += iIndex[i];
		}
		if (loop == 0 || tempnum == 0)
			break;
	}

	double chP = 1;
	double nchP = 1;

	//Determine if there are any remaining regions
	int lgLoc, lgNum;
	lgNum = 0;
	for (int i = 0; i < num; i++)
	{
		ch[i] = 0;
		nch[i] = 0;
		if (iIndex[i] > lgNum)
		{
			lgLoc = i;
			lgNum = iIndex[i];
		}
	}

	//If there are
	if (lgNum > 0)
	{
		if (ShowTables)
		{
			cout << "If there are remaining regions: " << endl;
			showtable(totalNum, num, state, t, camera, true);
		}
		loop = 1;
		while (loop)
		{
			ch[lgLoc] = 1;
			for (int k = 0; k < totalNum; k++)
			{
				if (t[lgLoc][k] == 1)
					for (int n = 0; n < num; n++)
						t[n][k] = 0;
			}

			if (ShowTables)
			{
				showtable(totalNum, num, state, t, camera, true);
			}

			for (int i = 0; i < num; i++)
				for (int j = 0; j < num; j++)
				{
					if (i != j)
					{
						if (removeRow(t[i], t[j], totalNum) == 1)
						{
							nch[i] = 1;
							for (int k = 0; k < totalNum; k++)
								t[i][k] = 0;
						}
						else if (removeRow(t[i], t[j], totalNum) == 2)
						{
							if (prob[i] < prob[j])
							{
								nch[i] = 1;
								for (int k = 0; k < totalNum; k++)
									t[i][k] = 0;
							}
						}
					}
				}

			if (ShowTables)
			{
				showtable(totalNum, num, state, t, camera, true);
			}

			for (int i = 0; i < num; i++)
			{
				kmarker = 0;
				imarker = 0;
				for (int j = 0; j < totalNum; j++)
				{
					if (t[i][j] == 1)
					{
						imarker = 1;
						for (int k = 0; k < num; k++)
							if ((t[k][j] == 1) && (k != i))
							{
								kmarker = 1;
								break;
							}
						if (kmarker == 1)
							break;
					}
				}
				if (kmarker == 0 && imarker == 1)
				{
					for (int j = 0; j < totalNum; j++)
						t[i][j] = 0;
					ch[i] = 1;
				}
			}

			if (ShowTables)
			{
				for (int i = 0; i < num; i++)
					cout << ch[i] << " ";
				cout << endl;
				for (int i = 0; i < num; i++)
					cout << nch[i] << " ";
				cout << endl;

				cout << "Select the group wich have largtest joint likelihood" << endl;
			}
			chP = 1;
			nchP = 1;

			for (int i = 0; i < num; i++)
			{
				if (ch[i] == 1)
				{
					if (ShowTables)
					{
						cout << "No." << i << ": " << prob[i] << endl;
					}
					chP = chP * prob[i];
				}
			}
			if (ShowTables)
			{
				cout << "total: " << chP << endl;

				cout << "VS" << endl;
			}

			for (int i = 0; i < num; i++)
			{
				if (nch[i] == 1)
				{
					if (ShowTables)
					{
						cout << "No." << i << ": " << prob[i] << endl;
					}
					nchP = nchP * prob[i];
				}
			}
			if (ShowTables)
			{
				cout << "total: " << nchP << endl;
			}


			if (nchP == 1 || nchP < chP)
			{
				for (int i = 0; i < num; i++)
					if (ch[i] == 1)
						state[i] = 1;
			}
			else
			{
				for (int i = 0; i < num; i++)
					if (nch[i] == 1)
						state[i] = 1;
			}

			showtable(totalNum, num, state, t, camera, true);

			for (int j = 0; j < num; j++)
				jmarker[j] = 0;

			for (int i = 0; i < totalNum; i++)
			{
				sum = 0;
				for (int j = 0; j < num; j++)
					sum += t[j][i];
				if (sum == 1)
				{
					for (int j = 0; j < num; j++)
						if (t[j][i] == 1)
							jmarker[j] = 1;
				}
			}

			for (int j = 0; j < num; j++)
			{
				if (jmarker[j] == 1)
				{
					for (int k = 0; k < totalNum; k++)
					{
						if (t[j][k] == 1)
							for (int n = 0; n < num; n++)
								t[n][k] = 0;
					}
					state[j] = 1;
				}
			}

			for (int i = 0; i < num; i++)
			{
				tempnum = 0;
				for (int j = 0; j < totalNum; j++)
				{
					tempnum += t[i][j];
				}
				if (iIndex[i] != tempnum)
				{
					iIndex[i] = tempnum;
				}
			}

			lgNum = 0;
			for (int i = 0; i < num; i++)
			{
				ch[i] = 0;
				nch[i] = 0;
				if (iIndex[i] > lgNum)
				{
					lgLoc = i;
					lgNum = iIndex[i];
				}
			}

			loop = 0;
			if (lgNum > 0)
			{
				loop = 1;
			}
		}
	}

	delete[] iIndex;
	delete[] jmarker;

	for (int i = 0; i < num; i++)
		delete[] t[i];
	delete[] t;

	//show results
	cout << "Result: ";
	for (int i = 0; i < num; i++)
	{
		_itoa(i, text, 10);

		if (state[i] == 1)
			cout << i << " ";

		if (i < 16)
		{
			if (state[i] != 1)
			{
				if (ShowPhantoms)
				{
					for (int k = 0; k < camera.size(); k++)
					{
						CvBlob rectb;
						if (st->at<uchar>(0, i) >> k & 0x01)
						{
							rectb.x = camera[k]->r[i].x + camera[k]->r[i].width / 2;
							rectb.y = camera[k]->r[i].y + camera[k]->r[i].height / 2;
							rectb.w = camera[k]->r[i].width;
							rectb.h = camera[k]->r[i].height;

							drawDashRect(&IplImage(camera[k]->frame), 1, 4, &rectb, colorful[i], 1);
						}

						cvPutText(&IplImage(camera[k]->frame), text, cvPoint(20 * i, 250), &font, colorful[i]);
					}						
				}
			}
			else
			{
				for (int k = 0; k < camera.size(); k++)
					cvPutText(&IplImage(camera[k]->frame), text, cvPoint(20 * i, 250), &font, colorful[i]);
				result.push_back(i);
				rstnum.push_back(i);
				for (int k = 0; k < camera.size(); k++)
					if (st->at<uchar>(0, i) >> k & 0x01)
						rectangle(camera[k]->frame, camera[k]->r[i], colorful[i], 2);
			}
		}
		else
		{
			if (state[i] != 1)
			{
				if (ShowPhantoms)
				{
					for (int k = 0; k < camera.size(); k++)
					{
						CvBlob rectb;
						if (st->at<uchar>(0, i) >> k & 0x01)
						{
							rectb.x = camera[k]->r[i].x + camera[k]->r[i].width / 2;
							rectb.y = camera[k]->r[i].y + camera[k]->r[i].height / 2;
							rectb.w = camera[k]->r[i].width;
							rectb.h = camera[k]->r[i].height;

							drawDashRect(&IplImage(camera[k]->frame), 1, 4, &rectb, colorful[i], 1);
						}
						cvPutText(&IplImage(camera[k]->frame), text, cvPoint(20 * (i - 16), 270), &font, colorful[i]);
					}
				}
			}
			else
			{
				for (int k = 0; k < camera.size(); k++)
					cvPutText(&IplImage(camera[k]->frame), text, cvPoint(20 * (i - 16), 270), &font, colorful[i]);
				result.push_back(i);
				rstnum.push_back(i);
				for (int k = 0; k < camera.size(); k++)
					if (st->at<uchar>(0, i) >> k & 0x01)
						rectangle(camera[k]->frame, camera[k]->r[i], colorful[i], 2);
			}
		}
	}
	cout << endl;
}

void QMmap(CvRect rect, Mat* pFrame, int i)
{
	int js = rect.y;
	int ks = rect.x;
	int je = rect.y + rect.height;
	int ke = rect.x + rect.width;
	int add = pow(2, i);

	for (int k = ks; k < ke; k++)
	{
		for (int j = js; j < je; j++)
		{
			pFrame->at<unsigned long long>(j, k) = pFrame->at<unsigned long long>(j, k) + add;
		}
	}
}

int RemoveDuplates(Mat* A, Mat*B, Mat* fmask)
{
	int k = 0;
	int lastNumber = 0;
	int mark = 0;
	int count = 0;
	int kk;

	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if (lastNumber != A->at<unsigned long long>(i, j))
			{
				mark = 0;
				for (int n = 0; n <= k; n++)
				{
					if (B->at<unsigned long long>(n, 0) == lastNumber)
					{
						B->at<unsigned long long>(n, 1) = B->at<unsigned long long>(n, 1) + count;
						count = 0;
						mark = 1;
						break;
					}
				}
				if (mark == 0)
				{
					k++;
					B->at<unsigned long long>(k, 0) = lastNumber;
					B->at<unsigned long long>(k, 1) = B->at<unsigned long long>(k, 1) + count;
					count = 0;
				}
			}
			lastNumber = A->at<unsigned long long>(i, j);
			count++;
		}
	}

	for (int n = 0; n <= k; n++)
		if (B->at<unsigned long long>(n, 0) == lastNumber)
			B->at<unsigned long long>(n, 1) = B->at<unsigned long long>(n, 1) + count;

	kk = k;

	lastNumber = 0;
	mark = 0;
	count = 0;

	for (int i = 0; i < A->rows; i++)
	{
		for (int j = 0; j < A->cols; j++)
		{
			if (fmask->at<uchar>(i, j) > 0)
			{
				if (lastNumber != A->at<unsigned long long>(i, j))
				{
					for (int n = 0; n <= k; n++)
					{
						if (B->at<unsigned long long>(n, 0) == lastNumber)
						{
							B->at<unsigned long long>(n, 2) = B->at<unsigned long long>(n, 2) + count;
							count = 0;
							break;
						}
					}
				}
				lastNumber = A->at<unsigned long long>(i, j);
				count++;
			}
		}
	}

	for (int n = 0; n <= k; n++)
		if (B->at<unsigned long long>(n, 0) == lastNumber)
			B->at<unsigned long long>(n, 2) = B->at<unsigned long long>(n, 2) + count;
	return kk + 1;
}

int removeRow(bool a[], bool b[], int num)
{
	int zeroNum = 0;

	int oneNum = 0;
	int aOneNum = 0;

	for (int i = 0; i < num; i++)
	{
		switch (a[i])
		{
		case 1:
			aOneNum++;
			switch (b[i])
			{
			case 0:
				return 0;
				break;
			case 1:
				oneNum++;
				break;
			default:
				break;
			}
			break;
		case 0:
			zeroNum++;
			switch (b[i])
			{
			case 1:
				oneNum++;
				break;
			}
			break;
		}
	}
	if (zeroNum == num)
		return 0;
	else if (oneNum == aOneNum)
		return 2;
	else
		return 1;

}

void showtable(int totalNum, int num, bool state[], bool** t, vector<Camera*> camera, bool shortT = 0)
{
	if (shortT == 0)
	{
		cout << endl;
		cout << "\t\b\b\b";
		for (int k = 0; k < camera.size(); k++)
		{
			for (int j = 0; j < camera[k]->Tnum; j++)
				cout << k + 1;
		}
		cout << " ";
		cout << endl;


		for (int i = 0; i < num; i++)
		{
			cout << "I" << i << ":";
			if (state[i] == 1)
				cout << "o";
			cout << "\t\b\b\b";

			for (int j = 0; j < totalNum; j++)
				if (t[i][j] == 0)
					cout << "+";
				else
					cout << "X";
			cout << endl;
		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "\t\b\b\b";
		int n = 0;
		for (int k = 0; k < camera.size(); k++)
		{
			for (int j = 0; j < camera[k]->Tnum; j++)
			{
				if (countcolumn[n] == 1)
					cout << k + 1;
				n++;
			}
		}
		cout << " ";
		cout << endl;


		for (int i = 0; i < num; i++)
		{
			cout << "I" << i << ":";
			if (state[i] == 1)
				cout << "o";
			cout << "\t\b\b\b";

			for (int j = 0; j < totalNum; j++)
			{
				if (countcolumn[j] == 1)
				{
					if (t[i][j] == 0)
						cout << "+";
					else
						cout << "X";
				}
			}
			cout << endl;
		}
		cout << endl;
	}
}

double Gaussian1D(double dis)
{
	if (dis >= 0)
	{
		if (dis <= tableG[1][0])
			return 1 - tableG[0][0];
		else if (dis > tableG[1][15])
			return 1 - tableG[0][15];
		else
		{
			for (int i = 0; i < 15; i++)
				if ((dis > tableG[1][i]) && (dis <= tableG[1][i + 1]))
					return 1 - (tableG[0][i] - (dis - tableG[1][i]) / (tableG[1][i + 1] - tableG[1][i]) * (tableG[0][i] - tableG[0][i + 1]));
			return -1;
		}
	}
	else
		return -1;
}

void drawDashRect(CvArr* img, int linelength, int dashlength, CvBlob* blob, CvScalar color, int thickness)
{
	int w = cvRound(blob->w);//width
	int h = cvRound(blob->h);//height

	int tl_x = cvRound(blob->x - blob->w / 2);//top left x
	int tl_y = cvRound(blob->y - blob->h / 2);//top  left y

	int totallength = dashlength + linelength;
	int nCountX = w / totallength;//
	int nCountY = h / totallength;//

	CvPoint start, end;//start and end point of each dash

	//draw the horizontal lines
	start.y = tl_y;
	start.x = tl_x;

	end.x = tl_x;
	end.y = tl_y;

	for (int i = 0; i < nCountX; i++)
	{
		end.x = tl_x + (i + 1)*totallength - dashlength;//draw top dash line
		end.y = tl_y;
		start.x = tl_x + i*totallength;
		start.y = tl_y;
		cvLine(img, start, end, color, thickness);
	}
	for (int i = 0; i < nCountX; i++)
	{
		start.x = tl_x + i*totallength;
		start.y = tl_y + h;
		end.x = tl_x + (i + 1)*totallength - dashlength;//draw bottom dash line
		end.y = tl_y + h;
		cvLine(img, start, end, color, thickness);
	}

	for (int i = 0; i < nCountY; i++)
	{
		start.x = tl_x;
		start.y = tl_y + i*totallength;
		end.y = tl_y + (i + 1)*totallength - dashlength;//draw left dash line
		end.x = tl_x;
		cvLine(img, start, end, color, thickness);
	}

	for (int i = 0; i < nCountY; i++)
	{
		start.x = tl_x + w;
		start.y = tl_y + i*totallength;
		end.y = tl_y + (i + 1)*totallength - dashlength;//draw right dash line
		end.x = tl_x + w;
		cvLine(img, start, end, color, thickness);
	}
	start.x = tl_x + w;
	start.y = tl_y + h;
	end.x = tl_x + w;
	end.y = tl_y + h - 2;
	cvLine(img, start, end, color, thickness);

	end.x = tl_x + w - 2;
	end.y = tl_y + h;
	cvLine(img, start, end, color, thickness);
}

int dfs(int i)
{
	for (int j = 0; j < nd; j++)
	{
		if (!visit[j] && match[i][j])
		{
			visit[j] = 1;
			if (mark[j] == -1 || dfs(mark[j]))
			{
				mark[j] = i;
				return 1;
			}
		}
	}
	return 0;
}

int Hungary()
{
	int ans = 0;
	memset(mark, -1, sizeof(mark));
	for (int i = 0; i < ng; i++)
	{
		memset(visit, 0, sizeof(visit));
		if (dfs(i))
		{
			ans++;
		}
	}
	return ans;
}