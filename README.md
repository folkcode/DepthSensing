# Depth Sensing
This program is a collection of depth sensing projects using the Xtion2 depth camera.

# Dependencies
* OpenNI2/Xtion2 SDK. Download [here.](https://www.asus.com/3D-Sensor/Xtion-2/HelpDesk_Download/)
* OpenCV
* oscpack

# Building
* Use the included CMakelists.txt with Cmake to build.
* Binaries will be inside build/Release or build/Debug depending on build configurations.

# Projects
## depth_capture
Base project to capture and display depth images.

## depth_filtering
Filters objects farther than a threshold range. Objects closer than this range will be detected using OpenCV's blob detection.
The result points are transmitted with OSC so other applications can use them.
### Data Format
When sending through OSC, the data format is as follows :
x1, y1, x2, y2, ... , end

You can check for the string "end" to determine end of data.
