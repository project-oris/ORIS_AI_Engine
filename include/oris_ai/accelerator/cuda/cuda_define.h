/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute (ETRI)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *******************************************************************************/
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

/**
 * @brief Macro for implementing grid-stride loops in CUDA kernels.
 * 
 * This macro creates a for loop that allows each thread to process multiple elements
 * in a grid-stride pattern, ensuring efficient parallel processing across all available
 * threads.
 * 
 * @param i The loop variable name
 * @param n The total number of elements to process
 */
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
