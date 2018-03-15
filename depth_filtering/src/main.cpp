#include <OpenNI.h>
#include <opencv2\opencv.hpp>
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>

using namespace cv;
void main()
{
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

		while (1) {
			int changedIndex;

			openni::OpenNI::waitForAnyStream(&streams[0], streams.size(), &changedIndex);
			if (changedIndex == 0) {
				openni::VideoFrameRef depthFrame;
				depthStream.readFrame(&depthFrame);
				if (depthFrame.isValid()) {
					depthImage = cv::Mat(depthStream.getVideoMode().getResolutionY(),
						depthStream.getVideoMode().getResolutionX(),
						CV_16U, (char*)depthFrame.getData());

					// 0-10000mmまでのデータを0-255にする
					depthImage.setTo(0, depthImage > 2500);
					depthImage.convertTo(depthImage, CV_8U, 255.0 / 2500);

					// 中心点の距離を表示する
					auto videoMode = depthStream.getVideoMode();
					int centerX = videoMode.getResolutionX() / 2;
					int centerY = videoMode.getResolutionY() / 2;
					int centerIndex = (centerY * videoMode.getResolutionX()) + centerX;
					short* depth = (short*)depthFrame.getData();

					std::stringstream ss;
					ss << "Center Point :" << depth[centerIndex];
					cv::putText(depthImage, ss.str(), cv::Point(0, 50),
						cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 1);
					cv::imshow("Depth Camera", depthImage);
				}
			}

			int key = cv::waitKey(10);
			if (key == 'q') {
				break;
			}
		}
	}
	catch (std::exception&) {
		std::cout << openni::OpenNI::getExtendedError() << std::endl;
	}
}