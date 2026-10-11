# OpenCV: host image operators (core, imgproc), rotated-box geometry (imgproc in
# OpenCV 4, the geometry module in OpenCV 5) and image files for the examples
# (imgcodecs). Point CMake at an installation with -DOpenCV_DIR=<prefix>/lib/cmake/opencv4.

find_package(OpenCV REQUIRED COMPONENTS core imgproc OPTIONAL_COMPONENTS imgcodecs geometry PATHS ${OpenCV_DIR})
message(STATUS "Found OpenCV ${OpenCV_VERSION}: ${OpenCV_DIR}")
