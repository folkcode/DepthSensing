# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(depth_image)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set( OpenCV_DIR "D:/Libraries/opencv3.4.0/build" )
option( OpenCV_STATIC OFF )
find_package(OpenCV REQUIRED)

# URG sensor
set(XTION_ROOT                 "D:/Applications/Xtion2/Installation32" )
set(XTION_INCLUDE_DIRS         "D:/Applications/Xtion2/Installation32/Include")
set(XTION_LIB_RELEASE_DIR      "D:/Applications/Xtion2/Installation32/Lib")

set(OPENNI_LIBS_RELEASE  ${XTION_LIB_RELEASE_DIR}/OpenNI2.lib)

# OSCPACK
set(OSCPACK_INCLUDE_DIR      "D:/Libraries/oscpack_1_1_0")

# Set installation folder
set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "default install path" FORCE )
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build CACHE PATH "Build directory" FORCE)

# Set additional include folders
set (ADDITIONAL_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/src/include")

include_directories(${ADDITIONAL_INCLUDE})
include_directories(${XTION_INCLUDE_DIRS} ${OSCPACK_INCLUDE_DIR})
link_directories(${XTION_LIB_RELEASE_DIR})

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OSCPACK libraries : ${OSCPACK_INCLUDE_DIR}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


# Declare the executable target built from your sources
set(SOURCES  src/main.cpp)
add_executable(depth_image ${SOURCES} ${ADDITIONAL_INCLUDE}/OniSampleUtilities.h)

if(WIN32)
  target_link_libraries(depth_image wsock32 ws2_32 winmm setupapi)
endif()

# Link your application with OpenCV libraries
target_link_libraries(depth_image ${OpenCV_LIBS} ${OPENNI_LIBS_RELEASE})