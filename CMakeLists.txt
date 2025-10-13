cmake_minimum_required(VERSION 3.10)

project(ORIS_AI_OSS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/oris_ai/protobuf)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(LIB_ARCH_DIR "${PROJECT_SOURCE_DIR}/lib/aarch_64")
  message(STATUS "Building for aarch64 architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(FATAL_ERROR "Building for x86_64 architecture is currently not supported due to library compatibility issues.")
else()
  message(STATUS "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

link_directories(${LIB_ARCH_DIR})
message(STATUS "Library directory set to: ${LIB_ARCH_DIR}")
set(ORIS_AI_LIB "ORISCore")
set(ACCELERATOR_CUDA_Kernel_LIB "ORISCUDA")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(target_libraries "")

include(${PROJECT_SOURCE_DIR}/cmake/findCUDA.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/Packages.cmake)

file(GLOB EXAMPLES_SRCS src/oris_ai/examples/*.cc)

foreach(EXAMPLE_SRC ${EXAMPLES_SRCS})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)

  add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})

  target_link_libraries(${EXAMPLE_NAME} ${ORIS_AI_LIB} ${ACCELERATOR_CUDA_Kernel_LIB} ${target_libraries})
endforeach()
