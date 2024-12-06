cmake_minimum_required(VERSION 3.10)

project(ORIS_AI_OSS)

# C++17 표준 사용
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 경고와 최적화 옵션 설정
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")

# Build type 설정
set(CMAKE_BUILD_TYPE Release)

# 출력 디렉토리 설정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include 디렉토리 설정
include_directories(${PROJECT_SOURCE_DIR}/include)

# 시스템 아키텍처 확인
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(LIB_ARCH_DIR "${PROJECT_SOURCE_DIR}/lib/x86_64")
  message(STATUS "Building for x86_64 architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(LIB_ARCH_DIR "${PROJECT_SOURCE_DIR}/lib/aarch_64")
  message(STATUS "Building for aarch64 architecture.")
else()
  message(STATUS "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Lib 디렉토리 및 라이브러리 설정
link_directories(${LIB_ARCH_DIR})
message(STATUS "Library directory set to: ${LIB_ARCH_DIR}")
set(ORIS_AI_LIB "ORISCore")
set(ACCELERATOR_CUDA_Kernel_LIB "ORISCUDA")
set(ORIS_AI_OSS_LIB "ORISOSS")

# 필수 설치 패키지
set(target_libraries "")
include(${PROJECT_SOURCE_DIR}/cmake/Packages.cmake)

####################################################################
# examples 디렉토리에서 사용할 실행 파일을 설정하고 ORIS_AI_LIBS를 링크
file(GLOB EXAMPLES_SRCS src/oris_ai/examples/*.cc)

foreach(EXAMPLE_SRC ${EXAMPLES_SRCS})
  get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)

  # 예시 파일을 빌드
  add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})

  # 예제에 라이브러리 연결
  target_link_libraries(${EXAMPLE_NAME} ${ORIS_AI_LIB} ${ACCELERATOR_CUDA_Kernel_LIB} ${ORIS_AI_OSS_LIB} ${target_libraries})
endforeach()
