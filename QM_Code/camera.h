#pragma once
#include "BackgroundSubtractorSuBSENSE.h"
#include "cameraModel.h"
#include <opencv2/opencv.hpp> 
#include <string>
#include <stdio.h>
#include<vector>
using namespace std;
using namespace cv;

class Camera
{
public:
	Etiseo::CameraModel *cam;
	cv::Mat fore;
	cv::Mat map;
	cv::Mat maph;
	cv::Mat top;
	cv::Mat topshow;
	cv::Mat frame;
	cv::Mat foreground;
	cv::Mat rctMap;
	cv::BackgroundSubtractorMOG2 mog;
	VideoCapture capture;
	int FrameNumber;

	bool flag;
	Rect r[400];

	Mat pF;
	Mat T;
	int Tnum;
	double prob[400];
	double probB[400];
	double probT[400];
	int idx[400];
	Point p[400];

	Mat foreground3;
	BackgroundSubtractorSuBSENSE oBGSAlg;

	double tempSum = 0;
	double tempMax = 0;
	double temF;
	double temB;
	
	//down sample ratio of original frame
	int downsample;
	//scale from world coordinate to top view coordinate
	int scale;
	//shift from world coordinate to top view coordinate
	int shift;
	//height of boxes
	int height;
	//ratio of width and height
	double WHratio;

	//variable use for calculate duty cycle
	Mat iiimage;
	double temphistvalue;

	Camera(int Cdownsample, int Cscale, int Cshift, int Cheight, double CWHratio, string videoLocation, string bgLocation = "")
	{
		downsample = Cdownsample;
		scale = Cscale;
		shift = Cshift;
		height = Cheight;
		WHratio = CWHratio;

		FrameNumber = 0;
		cam = new Etiseo::CameraModel;

		capture.open(videoLocation);
		if (!capture.isOpened())
			cout << "Fail to open!  " << videoLocation << endl;

		if (bgLocation == "")
		{
			cout << "Warning: No background figure. Initial background model with the first frame." << endl;
			capture.read(frame);
			capture.set(CV_CAP_PROP_POS_FRAMES, 0);
		}
		else
		{
			capture.read(frame);
		}

		top = Mat::zeros(1000, 1000, CV_8UC1);
		topshow = Mat::zeros(1000, 1000, CV_8UC1);

		map = Mat::zeros(1000, 1000, CV_32FC2)*(-1);
		maph = Mat::ones(1000, 1000, CV_32FC2)*(-1);

		rctMap = Mat::zeros(1000, 1000, CV_32FC4);

		pF = Mat::zeros(frame.rows, frame.cols, CV_64F);
		T = Mat::zeros(10000, 3, CV_64F);
	}

	void initialBG(bool flag1 = 0)  //1 for subsense, 0 for Gauss
	{
		flag = flag1;
		if (flag)
		{
			cv::Mat oSequenceROI(frame.size(), CV_8UC1, cv::Scalar_<uchar>(255));
			oBGSAlg.initialize(frame, oSequenceROI);
			oBGSAlg(frame, fore, double(FrameNumber <= 100));
			fore.copyTo(foreground);
		}
		else
		{
			mog(frame, foreground, 1);
		}
	}

	void readNextFrame()
	{
		capture.read(frame);
		FrameNumber++;
	}

	void setFrameNumber(int n)
	{
		capture.set(CV_CAP_PROP_POS_FRAMES, n);
		FrameNumber = n;
	}

	void readFrameNumber(int n)
	{
		capture.set(CV_CAP_PROP_POS_FRAMES, n);
		capture.read(frame);
		capture.set(CV_CAP_PROP_POS_FRAMES, FrameNumber);
	}

	void updateMog(double rate = 0.0005)
	{
		if (flag)
		{
			if (FrameNumber <= 50)
				oBGSAlg(frame, fore, double(FrameNumber <= 50));
			else
				oBGSAlg(frame, fore, 0);
			foreground.release();
			fore.copyTo(foreground);
		}
		else
		{
			mog(frame, fore, rate);
			foreground.release();
			fore.copyTo(foreground);
			threshold(foreground, foreground, 250, 255, cv::THRESH_BINARY);
		}
	}

	void mapToTop()
	{
		double X, Y, Z = 0;
		double x, y;
		double xt, yt;
		double a, b, c, d, e, f;
		double aa, bb, cc, dd;
		bool flag1, flag2, flag3, flag4;
		for (int i = 0; i < 1000; i = i + 1)
		{
			for (int j = 0; j < 1000; j = j + 1)
			{
				X = (i - shift) * scale;
				Y = (j - shift) * scale;
				Z = 0;
				cam->worldToImage(X, Y, Z, x, y);
				x = x / downsample;
				y = y / downsample;
				Z = height;
				cam->worldToImage(X, Y, Z, xt, yt);
				xt = xt / downsample;
				yt = yt / downsample;
				a = x - abs(y - yt)*WHratio*0.5;
				b = yt;
				c = abs(y - yt)*WHratio;
				d = y - yt;
				e = a + c;
				f = b + d;
				Z = 1200; //waist plan height
				cam->worldToImage(X, Y, Z, xt, yt);
				xt = xt / downsample;
				yt = yt / downsample;

				if (x > 0 && y > 0 && x < foreground.cols&&y < foreground.rows)
					topshow.at<uchar>(i, j) = 1;

				if (d > 0)
				{
					flag1 = (a > 0 && a < foreground.cols) && (b > 0 && b < foreground.rows);
					flag2 = (b > 0 && b < foreground.rows) && (e > 0 && e < foreground.cols);
					flag3 = (a > 0 && a < foreground.cols) && (f > 0 && f < foreground.rows);
					flag4 = (f > 0 && f < foreground.rows) && (e > 0 && e < foreground.cols);
					if (flag1 || flag2 || flag3 || flag4)
					{
						top.at<uchar>(i, j) = 1;

						map.at<Point2f>(i, j).x = x;
						map.at<Point2f>(i, j).y = y;

						if (xt < 0)
							xt = 0;
						else if (xt> foreground.cols)
							xt = foreground.cols;

						if (yt < 0)
							yt = 0;
						else if (yt> foreground.rows)
							yt = foreground.rows;

						maph.at<Point2f>(i, j).x = xt;
						maph.at<Point2f>(i, j).y = yt;
					}
				}

				aa = a > 0 ? a : 0;
				bb = b > 0 ? b : 0;
				if (a < 0)
					c = c + a;
				cc = a + c < foreground.cols ? c : (foreground.cols - a);
				dd = b + d < foreground.rows ? d : (foreground.rows - b);
				rctMap.at<Vec4f>(i, j)[0] = aa;
				rctMap.at<Vec4f>(i, j)[1] = bb;
				rctMap.at<Vec4f>(i, j)[2] = cc;
				rctMap.at<Vec4f>(i, j)[3] = dd;
			}
		}
	}

	double dutyCycle(int i, int j)
	{
		if ((topshow.at<uchar>(i, j) > 0))
		{
			tempMax = 0;
			tempSum = 0;

			for (int m = 0; m < rctMap.at<Vec4f>(i, j)[2]; m++)
			{
				temphistvalue = iiimage.at<int>(rctMap.at<Vec4f>(i, j)[1] + rctMap.at<Vec4f>(i, j)[3], rctMap.at<Vec4f>(i, j)[0] + m + 1) + iiimage.at<int>(rctMap.at<Vec4f>(i, j)[1], rctMap.at<Vec4f>(i, j)[0] + m)
					- iiimage.at<int>(rctMap.at<Vec4f>(i, j)[1], rctMap.at<Vec4f>(i, j)[0] + m + 1) - iiimage.at<int>(rctMap.at<Vec4f>(i, j)[1] + rctMap.at<Vec4f>(i, j)[3], rctMap.at<Vec4f>(i, j)[0] + m);
				temF = 1 - 3.0 * abs(m - rctMap.at<Vec4f>(i, j)[2] / 2.0) / (double)rctMap.at<Vec4f>(i, j)[2] > 0 ? 1 - 3.0 * abs(m - rctMap.at<Vec4f>(i, j)[2] / 2.0) / (double)rctMap.at<Vec4f>(i, j)[2] : 0;
				temB = abs(1 - 3.0 * abs(m - rctMap.at<Vec4f>(i, j)[2] / 2.0) / (double)rctMap.at<Vec4f>(i, j)[2] < 0 ? 1 - 3.0 * abs(m - rctMap.at<Vec4f>(i, j)[2] / 2.0) / (double)rctMap.at<Vec4f>(i, j)[2] : 0);
				tempSum += temphistvalue * temF;
				tempSum += (rctMap.at<Vec4f>(i, j)[3] - temphistvalue) * temB;
				tempMax += temF + temB;
			}
			tempMax *= rctMap.at<Vec4f>(i, j)[3];

			return tempSum / tempMax;
		}
		else
			return 1;
	}


private:


};