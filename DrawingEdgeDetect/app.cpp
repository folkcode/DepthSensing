#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
void EMSegmentation(Mat image, int no_of_clusters = 2);
void Mask(Mat input, Mat mask);
void DetectSquares(Mat input);
double median(cv::Mat channel);
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);
vector<Point> contoursConvexHull(vector<vector<Point> > contours);

Mat detectCanny(Mat input);

Rect DetectContours(Mat input, int drawFlag, int convexFlag, int roiFlag);
int main(int argc, const char** argv)
{
	Mat img = imread("nurie_crop.JPG", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
	if (img.empty()) //check whether the image is loaded or not
	{
		cout << "Error : Image cannot be loaded..!!" << endl;
		//system("pause"); //wait for a key press
		return -1;
	}

	namedWindow("MyWindow", CV_WINDOW_NORMAL); //create a window with the name "MyWindow"
	namedWindow("copy", CV_WINDOW_NORMAL);
	Rect imgRoi = DetectContours(img, 1, 0, 1);
	Mat zoomedImg = img(imgRoi);
	//Mask(img, mask);
	EMSegmentation(img, 2);
	imshow("MyWindow", img); //display the image which is stored in the 'img' in the "MyWindow" window
	imshow("copy", zoomedImg);
	waitKey(0); //wait infinite time for a keypress

	destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"

	return 0;
}

void DetectSquares(Mat input) {
	cv::Mat dst;
	std::vector<std::vector<cv::Point>> contours;
	// Convert to grayscale
	Mat gray; Mat bw;
	cv::cvtColor(input, gray, CV_BGR2GRAY);
	blur(gray, bw, Size(3, 3));
	cv::Canny(gray, bw, 80, 240, 3);
	cv::findContours(bw.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	std::vector<Point> approx;
	input.copyTo(dst);
	cv::Rect bounding_rect;

	for (int i = 0; i < contours.size(); i++)
	{
		bounding_rect = boundingRect(contours[i]);
		// Approximate contour with accuracy proportional
		// to the contour perimeter
		cv::approxPolyDP(cv::Mat(contours[i]), approx, cv::arcLength(cv::Mat(contours[i]), true)*0.02, true);

		// Skip small or non-convex objects
		if (std::fabs(cv::contourArea(contours[i])) < 100 || !cv::isContourConvex(approx))
			continue;

		if (approx.size() >= 4 && approx.size() <= 6)
		{
			// Number of vertices of polygonal curve
			int vtc = approx.size();

			// Get the cosines of all corners
			std::vector<double> cos;
			for (int j = 2; j < vtc + 1; j++)
				cos.push_back(angle(approx[j%vtc], approx[j - 2], approx[j - 1]));

			// Sort ascending the cosine values
			std::sort(cos.begin(), cos.end());

			// Get the lowest and the highest cosine
			double mincos = cos.front();
			double maxcos = cos.back();

			// Use the degrees obtained above and the number of vertices
			// to determine the shape of the contour
			//if (vtc == 4) {
			rectangle(input, bounding_rect, Scalar(0, 255, 0), 2, 8, 0);
			//}

		}
	}
}

/**
* Helper function to find a cosine of angle between vectors
* from pt0->pt1 and pt0->pt2
*/
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
	double dx1 = pt1.x - pt0.x;
	double dy1 = pt1.y - pt0.y;
	double dx2 = pt2.x - pt0.x;
	double dy2 = pt2.y - pt0.y;
	return (dx1*dx2 + dy1*dy2) / sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

Rect DetectContours(Mat input, int drawFlag, int convexFlag, int roiFlag) {
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;
	Mat thr;
	Mat thr_roi_gray;;
	Mat copy = input.clone();
	cvtColor(input, thr, COLOR_BGR2GRAY); //Convert to gray
	threshold(thr, thr, 100, 120, THRESH_BINARY); //Threshold the gray
	threshold(thr, thr, 20, 255, CV_THRESH_BINARY_INV);
	vector<vector<Point> > contours; // Vector for storing contours
	std::vector<cv::Vec4i> hierarchy;
	findContours(thr, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image
	int levels = 2;
	for (size_t i = 0; i< contours.size(); i++) // iterate through each contour.
	{
		double area = contourArea(contours[i]);  //  Find the area of contour
												 // look for hierarchy[i][3]!=-1, ie hole boundaries
		if (hierarchy[i][3] != -1 && drawFlag == 1) {
			// random colour
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(input, contours, i,  colour,7,8,noArray(),2,Point(0,0));
		}
		if (area > largest_area)
		{
			largest_area = area;
			largest_contour_index = i;               //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}

	//drawContours(input, contours, largest_contour_index, Scalar(0, 255, 0), 2); // Draw the largest contour using previously stored index.
	if (convexFlag == 1) {
		vector<Point> ConvexHullPoints = contoursConvexHull(contours);
		polylines(input, ConvexHullPoints, true, Scalar(0, 0, 255), 7);
	}
	rectangle(input, bounding_rect, Scalar(255, 0, 0), 5, 8, 0);
	
	if (roiFlag == 1) {
		cv::Rect roi(bounding_rect.x + 110, bounding_rect.y + 110, bounding_rect.width - 220, bounding_rect.height - 220);
		return roi;
	}
	else {
		cv::Rect roi(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height);
		return roi;
	}
	
}

void Mask(Mat input, Mat mask) {
	Mat dst;
	dst.setTo(Scalar(255, 255, 255));
	input.copyTo(dst, mask);
}

double median(cv::Mat channel)
{
	double m = (channel.rows*channel.cols) / 2;
	int bin = 0;
	double med = -1.0;

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;
	cv::Mat hist;
	cv::calcHist(&channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	for (int i = 0; i < histSize && med < 0.0; ++i)
	{
		bin += cvRound(hist.at< float >(i));
		if (bin > m && med < 0.0)
			med = i;
	}

	return med;
}

Mat detectCanny(Mat input) {

	Mat input_gray;
	cvtColor(input, input_gray, COLOR_BGR2GRAY);
	Mat canny;

	Mat mCanny_Gray, mThres_Gray, result;
	double CannyAccThresh = threshold(input_gray, mThres_Gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	double CannyThresh = 0.1 * CannyAccThresh;

	int erosion_size = 1;
	Mat element = getStructuringElement(cv::MORPH_CROSS,
		cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		cv::Point(erosion_size, erosion_size));

	// Apply erosion or dilation on the image
	

	Canny(input_gray, canny, CannyThresh, CannyAccThresh);
	//morphologyEx(canny, canny, CV_MOP_GRADIENT, element, Point(-1, -1), 1, 1);
	dilate(canny, canny, element, Point(-1, -1), 30, 1);
	return canny;
}

vector<Point> contoursConvexHull(vector<vector<Point> > contours)
{
	vector<Point> result;
	vector<Point> pts;
	for (size_t i = 0; i< contours.size(); i++)
		for (size_t j = 0; j< contours[i].size(); j++)
			pts.push_back(contours[i][j]);
	convexHull(pts, result);
	return result;
}

/**
* Create a sample vector out of RGB image
*/
Mat asSamplesVectors(Mat& img) {
	//convert the input image to float
	cv::Mat floatSource;
	img.convertTo(floatSource, CV_32F);

	//now convert the float image to column vector
	cv::Mat samples(img.rows * img.cols, 3, CV_32FC1);
	int idx = 0;
	for (int y = 0; y < img.rows; y++) {
		cv::Vec3f* row = floatSource.ptr<cv::Vec3f >(y);
		for (int x = 0; x < img.cols; x++) {
			samples.at<cv::Vec3f >(idx++, 0) = row[x];
		}
	}
	return samples;
}

/**
Perform segmentation (clustering) using EM algorithm
**/
void EMSegmentation(Mat image, int no_of_clusters) {
	Mat samples = asSamplesVectors(image);

	cout << "Starting EM training" << endl;
	/*
	Ptr<cv::ml::EM> em = cv::ml::EM::create();
	em->setClustersNumber(no_of_clusters);
	
	Mat responses;
	Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(samples,
		cv::ml::ROW_SAMPLE, responses,
		noArray(),
		noArray(),
		noArray(),
		noArray()
	);

	em->train(trainData);
	*/

	cv::Ptr<cv::ml::EM> source_model = cv::ml::EM::create();
	source_model->setClustersNumber(no_of_clusters);
	cv::Mat logs;
	cv::Mat labels;
	cv::Mat probs;
	if (source_model->trainEM(samples, logs, labels, probs))
	{
		std::cout << "true train em";
		for (cv::MatIterator_<int> it(labels.begin<int>()); it != labels.end<int>(); it++)
		{
			std::cout << (*it) << std::endl; // int i = *it
		}
	}
	else {
		std::cout << "false train em" << std::endl;
	}
	for (;;) {
		if (waitKey(5) > 0) break;
	}
	cout << "Finished training EM" << endl;

}