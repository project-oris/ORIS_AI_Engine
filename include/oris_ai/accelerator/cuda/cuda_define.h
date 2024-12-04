/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include <stdio.h>   // for fprintf
#include <stdlib.h>  // for exit

#include <cuda_runtime.h>

/**
 * @brief Macro to check CUDA function calls for errors.
 * 
 * This macro evaluates a given CUDA function call and checks if it returns a 
 * success status. If an error occurs, the macro logs the error message.
 * 
 * @param condition The CUDA function call to be checked.
 */
#define CUDA_CHECK(condition)                                         \
  do {                                                                \
    cudaError_t error = (condition);                                  \
    if (error != cudaSuccess) {                                       \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

/**
 * @brief The number of threads per block in CUDA kernel launches.
 * 
 * This constant defines the standard number of threads to be used per block 
 * for CUDA operations. Adjusting this value may impact performance depending 
 * on the kernel's requirements.
 */
constexpr int CUDA_THREADS_PER_BLOCK = 256;

/**
 * @brief Computes the number of blocks required for a given number of threads.
 * 
 * This function calculates the necessary number of blocks based on the 
 * total number of threads (N) and the predefined number of threads per block 
 * (`CUDA_THREADS_PER_BLOCK`). It ensures that all threads are covered 
 * in the allocated blocks.
 * 
 * @param N The total number of threads for the CUDA operation.
 * @return The number of blocks required to accommodate all threads.
 */
inline int CUDA_GET_BLOCKS(const int N) {
  return (N + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
}