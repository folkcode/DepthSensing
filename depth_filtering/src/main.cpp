#include <OpenNI.h>
#include <opencv2\opencv.hpp>
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>

// oscpack
#include <ip/UdpSocket.h>
#include <osc/OscOutboundPacketStream.h>

using namespace cv;
using namespace std;

RNG rng(12345);
int thresh = 100;
int max_thresh = 255;

#define OUTPUT_BUFFER_SIZE 1024

const char* ip_address = "127.0.0.1";
unsigned int port = 5001;

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

void main()
{
	stringstream data_string;
	data_string.str(std::string());
	try {
		openni::OpenNI::initialize();
		openni::Device device;
		auto ret = device.open(openni::ANY_DEVICE);

		if (ret != openni::STATUS_OK) {
			throw std::runtime_error("");
		}

		openni::VideoStream depthStream;

		depthStream.create(device, openni::SensorType::SENSOR_DEPTH);
		depthStream.start();

		std::vector<openni::VideoStream*> streams;

		streams.push_back(&depthStream);
		cv::Mat depthImage;

		int erosion_size = 6;
		Mat element = getStructuringElement(cv::MORPH_DILATE,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));

		// Set up the detector with default parameters.
		SimpleBlobDetector::Params params;

		params.filterByInertia = false;
		params.filterByConvexity = false;
		params.filterByArea = true;
		params.minArea = 1000;
		params.maxArea = 10000;
		params.filterByColor = true;
		params.blobColor = 255;
		Ptr<SimpleBlobDetector> d = SimpleBlobDetector::create(params);

		UdpTransmitSocket transmitSocket(IpEndpointName(ip_address, port));

		while (1) {

			int changedIndex;

			// Clear the data string variable every frame;
			data_string.str(std::string());

			openni::OpenNI::waitForAnyStream(&streams[0], streams.size(), &changedIndex);
			if (changedIndex == 0) {
				openni::VideoFrameRef depthFrame;
				depthStream.readFrame(&depthFrame);
				if (depthFrame.isValid()) {
					depthImage = cv::Mat(depthStream.getVideoMode().getResolutionY(),
						depthStream.getVideoMode().getResolutionX(),
						CV_16U, (char*)depthFrame.getData());

					// 0-2000mm‚Ü‚Å‚Ìƒf[ƒ^‚ð0-255‚É‚·‚é
					depthImage.setTo(0, depthImage > 1500);
					depthImage.convertTo(depthImage, CV_8U, 255.0 / 1500);
					
					Mat threshImage;
					threshold(depthImage, threshImage, 30, 255, THRESH_BINARY);
					dilate(threshImage, threshImage, element);

					// Detect blobs.
					std::vector<KeyPoint> keypoints;
					d->detect(threshImage, keypoints);

					// Draw detected blobs as red circles.
					// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
					Mat im_with_keypoints;
					drawKeypoints(threshImage, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

					if (keypoints.size() >= 1) {
						for (int i = 0; i < keypoints.size(); i++) {
							Point2f point = keypoints[i].pt;
							data_string << to_string(point.x) << "," << to_string(point.y);
						}
					}

					char buffer[OUTPUT_BUFFER_SIZE];
					osc::OutboundPacketStream ps(buffer, OUTPUT_BUFFER_SIZE);
					string send_string = data_string.str();
					send_string += "end";
					
					char *toSendText = new char[send_string.size() + 1];
					std::strcpy(toSendText, send_string.c_str());
					transmitSocket.Send(toSendText, strlen(toSendText) + 1);
					cout << send_string.size() << endl;
					imshow("keypoints", im_with_keypoints);
				}
			}

			int key = cv::waitKey(10);
			if (key == 'q') {
				break;
			}
		}
		// TODO : close device and other cleanup
		depthStream.stop();
		depthStream.destroy();
		device.close();
		openni::OpenNI::shutdown();
	}
	catch (std::exception&) {
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
	}
}

