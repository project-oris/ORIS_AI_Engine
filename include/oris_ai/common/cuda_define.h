/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include <cuda_runtime.h>
#include <glog/logging.h>

/**
 * @brief Macro to check CUDA function calls for errors.
 * 
 * This macro evaluates a given CUDA function call and checks if it returns a 
 * success status. If an error occurs, the macro logs the error message.
 * 
 * @param condition The CUDA function call to be checked.
 */
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
