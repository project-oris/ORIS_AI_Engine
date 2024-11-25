# CUDA
if(USE_CUDA)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  list(APPEND target_libraries
    ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
    ${CUDAToolkit_LIBRARY_DIR}/libcurand.so
    ${CUDAToolkit_LIBRARY_DIR}/libcublas.so
  )
endif()

# glog 헤더 및 라이브러리 포함
find_package(Glog REQUIRED)
include_directories(${GLog_INCLUDE_DIRS})
list(APPEND target_libraries ${GLog_LIBRARIES})

# OpenCV
find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Found OpenCV (include: ${OpenCV_INCLUDE_DIRS})")
message(STATUS "Found OpenCV (library: ${OpenCV_LIBS})")
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND target_libraries ${OpenCV_LIBRARIES})

# Protobuf
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
list(APPEND target_libraries ${PROTOBUF_LIBRARIES})