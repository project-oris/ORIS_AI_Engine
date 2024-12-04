/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
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
