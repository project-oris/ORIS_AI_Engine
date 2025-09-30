# CUDA
include_directories(${CUDAToolkit_INCLUDE_DIRS})
list(APPEND target_libraries ${CUDAToolkit_LIBRARIES})

# Eigen
find_package(Eigen3 QUIET REQUIRED)
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "Found Eigen3 (include: ${EIGEN3_INCLUDE_DIR})")
include_directories(${EIGEN3_INCLUDE_DIR})
list(APPEND target_libraries Eigen3::Eigen)

# OpenBLAS
find_package(OpenBLAS REQUIRED)
include_directories(${OpenBLAS_INCLUDE_DIRS})
list(APPEND target_libraries ${OpenBLAS_LIBRARIES})

# OpenCV
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs dnn)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Found OpenCV (include: ${OpenCV_INCLUDE_DIRS})")
message(STATUS "Found OpenCV (library: ${OpenCV_LIBS})")
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND target_libraries ${OpenCV_LIBRARIES})

# glog
find_package(Glog REQUIRED)
include_directories(${GLog_INCLUDE_DIRS})
list(APPEND target_libraries ${GLog_LIBRARIES})

# Protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
list(APPEND target_libraries ${PROTOBUF_LIBRARIES})