# OpenCV: host image operators and rotated-box geometry (core, imgproc), plus
# image files for the examples (imgcodecs). CMake at an
# installation with -DOpenCV_DIR=<prefix>/lib/cmake/opencv4.

find_package(OpenCV REQUIRED COMPONENTS core imgproc OPTIONAL_COMPONENTS imgcodecs PATHS ${OpenCV_DIR})
message(STATUS "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_DIR}")
