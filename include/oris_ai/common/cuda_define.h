/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#ifndef INCLUDE_ORIS_AI_COMMON_CUDA_DEFINE_H_
#define INCLUDE_ORIS_AI_COMMON_CUDA_DEFINE_H_

#include <cuda_runtime.h>
#include <glog/logging.h>

// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

# endif   // INCLUDE_ORIS_AI_COMMON_CUDA_DEFINE_H_