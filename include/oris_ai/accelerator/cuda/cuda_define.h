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
