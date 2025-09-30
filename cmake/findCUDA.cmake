find_package(CUDAToolkit REQUIRED)

function(detect_installed_gpus out_variable)
  set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

  file(WRITE ${__cufile} "
    #include <cuda_runtime.h>
    #include <cstdio>
    int main() {
      int count = 0;
      if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;
      if (count == 0) return -1;
      for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop;
        if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
          std::printf(\"%d.%d \", prop.major, prop.minor);
      }
      return 0;
    }")

  find_program(CUDA_NVCC_EXECUTABLE nvcc
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES bin)

  if (NOT CUDA_NVCC_EXECUTABLE)
    message(FATAL_ERROR "Could not find NVCC. Ensure CUDA is installed and NVCC is in your PATH.")
  endif()

  execute_process(
    COMMAND "${CUDA_NVCC_EXECUTABLE}" "--run" "${__cufile}"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    RESULT_VARIABLE __nvcc_res
    OUTPUT_VARIABLE __nvcc_out
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (__nvcc_res EQUAL 0)
    set(${out_variable} "${__nvcc_out}" PARENT_SCOPE)
  else()
    message(WARNING "Failed to detect GPU architectures using NVCC.")
    set(${out_variable} "" PARENT_SCOPE)
  endif()
endfunction()

if(CUDAToolkit_FOUND)
    set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})

    find_library(CUDA_CUBLAS_LIBRARY cublas HINTS ${CUDAToolkit_LIBRARY_DIR})
    find_library(CUDA_CUBLASLT_LIBRARY cublasLt HINTS ${CUDAToolkit_LIBRARY_DIR})
    find_library(CUDA_CUDART_LIBRARY cudart HINTS ${CUDAToolkit_LIBRARY_DIR})
    find_library(CUDA_CURAND_LIBRARY curand HINTS ${CUDAToolkit_LIBRARY_DIR})

    find_library(CUDA_CUDNN_LIBRARY cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})

    if(NOT CUDA_CUDNN_LIBRARY)
        message(FATAL_ERROR "cuDNN library not found. Please ensure that cuDNN is installed and available in the CUDA toolkit library path.")
    endif()

    if(NOT CUDA_CUBLASLT_LIBRARY)
        message(FATAL_ERROR "cuBLASLt library not found. Please ensure that cuBLASLt is installed and available in the CUDA toolkit library path.")
    endif()

    message(STATUS "CUDA Include Directories: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUBLAS Library: ${CUDA_CUBLAS_LIBRARY}")
    message(STATUS "cuBLASLt Library: ${CUDA_CUBLASLT_LIBRARY}")
    message(STATUS "CUDART Library: ${CUDA_CUDART_LIBRARY}")
    message(STATUS "CURAND Library: ${CUDA_CURAND_LIBRARY}")
    message(STATUS "cuDNN Library: ${CUDA_CUDNN_LIBRARY}")

    list(APPEND CUDAToolkit_LIBRARIES ${CUDA_CUBLAS_LIBRARY} ${CUDA_CUBLASLT_LIBRARY} ${CUDA_CUDART_LIBRARY} ${CUDA_CURAND_LIBRARY} ${CUDA_CUDNN_LIBRARY})
    
    detect_installed_gpus(GPU_ARCHITECTURE)

    if(GPU_ARCHITECTURE MATCHES "^[0-9]+\\.[0-9]+$")
        string(REPLACE "." "" GPU_ARCH ${GPU_ARCHITECTURE})
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})  # CMAKE_CUDA_ARCHITECTURES 설정
        message(STATUS "Detected GPU Architecture: sm_${GPU_ARCH}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES 60)
        message(WARNING "Could not detect GPU architecture. Using default architecture sm_60.")
    endif()

    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -Wall")
    message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")
else()
    message(WARNING "CUDA Toolkit not found.")
endif()
