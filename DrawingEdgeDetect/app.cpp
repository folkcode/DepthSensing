#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
void Mask(Mat input, Mat mask);
void DetectSquares(Mat input);
double angle(cv::Point pt1, cv::Point pt2, cv::Point pt0);
void DetectContours(Mat input);
int main(int argc, const char** argv)
{
	Mat img = imread("nurie.JPG", CV_LOAD_IMAGE_UNCHANGED); //read the image data in the file "MyPic.JPG" and store it in 'img'
	Mat mask = imread("nurie_mask.JPG", CV_LOAD_IMAGE_UNCHANGED);
	if (img.empty()) //check whether the image is loaded or not
	{
		cout << "Error : Image cannot be loaded..!!" << endl;
		//system("pause"); //wait for a key press
		return -1;
	}

	namedWindow("MyWindow", CV_WINDOW_NORMAL); //create a window with the name "MyWindow"
	namedWindow("thr", CV_WINDOW_NORMAL);
	resizeWindow("thr", 612, 816);
	
	namedWindow("copy", CV_WINDOW_NORMAL);
	resizeWindow("copy", 612, 816);
	//DetectContours(img);
	Mask(img, mask);
	resizeWindow("MyWindow", 612, 816);
	imshow("MyWindow", img); //display the image which is stored in the 'img' in the "MyWindow" window
	imshow("thr", mask);
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

void DetectContours(Mat input) {
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;
	Mat thr;
	Mat thr_roi_gray;;
	Mat copy = input.clone();
	cvtColor(input, thr, COLOR_BGR2GRAY); //Convert to gray
	threshold(thr, thr, 100, 120, THRESH_BINARY); //Threshold the gray
	threshold(thr, thr, 20, 255, CV_THRESH_BINARY_INV);
	imshow("thr", thr);
	vector<vector<Point> > contours; // Vector for storing contours
	std::vector<cv::Vec4i> hierarchy;
	findContours(thr, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE); // Find the contours in the image

	for (size_t i = 0; i< contours.size(); i++) // iterate through each contour.
	{
		double area = contourArea(contours[i]);  //  Find the area of contour
												 // look for hierarchy[i][3]!=-1, ie hole boundaries
		if (hierarchy[i][3] != -1) {
			// random colour
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			drawContours(input, contours, i, colour);
		}
		if (area > largest_area)
		{
			largest_area = area;
			largest_contour_index = i;               //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}

	drawContours(input, contours, largest_contour_index, Scalar(0, 255, 0), 2); // Draw the largest contour using previously stored index.
	rectangle(input, bounding_rect, Scalar(255, 0, 0), 5, 8, 0);

	// Now use bounding Rect and get ROI area
	cv::Rect roi(bounding_rect.x + 20, bounding_rect.y + 20, bounding_rect.width - 40, bounding_rect.height - 40);
	Mat input_roi = input(roi);
	
	Mat input_roi_gray;
	cvtColor(input_roi, input_roi_gray, COLOR_BGR2GRAY);
	blur(input_roi_gray, input_roi_gray,  Size(3, 3));
	Canny(input_roi_gray, input_roi_gray, 10, 30, 3, false);

	

	Mat hsvImg; Mat thresholdImg;
	cvtColor(input_roi, hsvImg, COLOR_BGR2HSV);
	vector<Mat> channels;
	split(hsvImg, channels);

	// get the average hue value of the image
	Scalar threshValue = mean(channels[0]);
	double thresholdVal = threshValue[0];

	threshold(channels[0], thresholdImg, thresholdVal, 179.0, THRESH_BINARY_INV);

	blur(thresholdImg, thresholdImg, Size(5, 5));

	dilate(thresholdImg, thresholdImg, Mat(), Point(-1, -1),1);
	erode(thresholdImg, thresholdImg, Mat(), Point(-1, -1), 3);

	threshold(thresholdImg, thresholdImg, thresholdVal, 179.0, THRESH_BINARY);
	
	Mat foreground = input_roi.clone();
	foreground.setTo(Scalar(255, 255, 255));
	input_roi.copyTo(foreground, thresholdImg);
	imshow("copy", foreground);
}

void Mask(Mat input, Mat mask) {
	Mat dst;
	dst.setTo(Scalar(255, 255, 255));
	input.copyTo(dst, mask);
	imshow("copy", dst);
}
