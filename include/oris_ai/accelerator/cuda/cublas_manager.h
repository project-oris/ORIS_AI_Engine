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

#include <cublas_v2.h>

/**
 * @brief Macro to check cuBLAS function calls for errors.
 * 
 * This macro evaluates a given cuBLAS function call and checks if it returns a
 * success status. If an error occurs, the macro logs the error message.
 * 
 * @param condition The cuBLAS function call to be checked.
 */
#define CUBLAS_CHECK(condition)                                       \
  do {                                                                \
    cublasStatus_t status = condition;                                \
    if (status != CUBLAS_STATUS_SUCCESS) {                            \
      fprintf(stderr, "CUBLAS Error: %s\n",                           \
              oris_ai::cublasGetErrorString(status));                 \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

namespace oris_ai {

/**
 * @brief Returns a string describing the cuBLAS error code.
 * 
 * This function maps cuBLAS error codes to corresponding error messages.
 * @param error The cuBLAS error code.
 * @return const char* Error message as a string.
 */
const char* cublasGetErrorString(cublasStatus_t error);

// class CUBLASManager {
//   public:
//     /**
//      * @brief Default constructor for CUBLASManager
//      * Initializes the CUBLAS handle
//      */
//     CUBLASManager();

//     /**
//      * @brief Destructor for CUBLASManager
//      * Cleans up the CUBLAS handle
//      */
//     ~CUBLASManager();

//     /**
//      * @brief Deleted copy constructor to prevent copying of the singleton instance.
//      */
//     CUBLASManager(const CUBLASManager&) = delete;

//     /**
//      * @brief Deleted copy assignment operator to prevent assignment of the singleton instance.
//      */
//     CUBLASManager& operator=(const CUBLASManager&) = delete;

//     /**
//      * @brief Singleton instance getter for CUBLASManager
//      * Ensures only one instance of the stream manager exists
//      * @return Reference to the singleton instance
//      */
//     static CUBLASManager& GetInstance() {
//       static CUBLASManager instance;
//       return instance;
//     }

//     /**
//      * @brief Getter for the CUBLAS handle
//      * @return cublasHandle_t cublas_handle
//      */
//     inline cublasHandle_t GetCUBLASHandle() {return cublas_handle_;}

//   protected:
//     cublasHandle_t cublas_handle_;
// };

} // namespace oris_ai