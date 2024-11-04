# OpenCV
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Found OpenCV (include: ${OpenCV_INCLUDE_DIRS})")
message(STATUS "Found OpenCV (library: ${OpenCV_LIBS})")
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND target_libraries ${OpenCV_LIBRARIES})