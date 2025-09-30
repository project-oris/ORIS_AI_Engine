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

#include <cudnn.h>

/**
 * @brief Macro to check cuDNN function calls for errors.
 * 
 * This macro evaluates a given cuDNN function call and checks if it returns a 
 * success status. If an error occurs, the macro logs the error message.
 * 
 * @param condition The cuDNN function call to be checked.
 */
#define CUDNN_CHECK(condition)                                           \
  do {                                                                   \
    cudnnStatus_t error = (condition);                                   \
    if (error != CUDNN_STATUS_SUCCESS) {                                 \
      fprintf(stderr, "cuDNN error: %s\n", cudnnGetErrorString(error));  \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CUDNN_VERSION_MIN(major, minor, patch) \
  (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

// class CUDNNManager {
//   public:
//     CUDNNManager() : use_benchmark_(true) {}
//     ~CUDNNManager() = default;

//     /**
//      * @brief Deleted copy constructor to prevent copying of the singleton instance.
//      */
//     CUDNNManager(const CUDNNManager&) = delete;

//     /**
//      * @brief Deleted copy assignment operator to prevent assignment of the singleton instance.
//      */
//     CUDNNManager& operator=(const CUDNNManager&) = delete;

//     /**
//      * @brief Singleton instance getter for CUDNNManager
//      * Ensures only one instance of the CUDNNManager exists
//      * @return Reference to the singleton instance
//      */
//     static CUDNNManager& GetInstance() {
//       static CUDNNManager instance;
//       return instance;
//     }

//     inline void SetUseBenchmark(bool use_benchmark) {use_benchmark_ = use_benchmark;}
//     inline bool GetUseBenchmark() {return use_benchmark_;}

//   protected:
//     bool use_benchmark_;
// };
