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

#include "oris_ai/layer/matmul.h"
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>

namespace oris_ai {

/**
 * @class MatMulGPU
 * @brief A class that implements matrix multiplication operations on a NVIDIA GPU.
 * 
 * This class provides a NVIDIA GPU-specific implementation for performing matrix 
 * multiplication operations, inheriting from the base `MatMul` class. It overrides 
 * the `Forward` method to perform the forward pass of the matrix multiplication 
 * layer using NVIDIA GPU resources with cuBLASLt optimization.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class MatMulGPU : public MatMul<T> {
  public:
    /**
     * @brief Constructor to initialize a MatMulGPU layer.
     * @param layer_name The name of the layer.
     */
    MatMulGPU(const std::string& layer_name);

    /**
     * @brief Destructor for the MatMulGPU class.
     */
    ~MatMulGPU();

    /**
     * @brief Overrides the virtual InitMatMul function for NVIDIA GPU-specific initialization.
     * 
     * This function implements the virtual InitMatMul method defined in the MatMul 
     * base class, configuring the matrix multiplication layer with the provided 
     * parameters for efficient execution on the NVIDIA GPU using cuBLASLt.
     * 
     * @param m The number of rows in matrix A (or transposed A).
     * @param n The number of columns in matrix B (or transposed B).
     * @param k The number of columns in matrix A (or transposed A) and rows in matrix B (or transposed B).
     * @param trans_a Flag indicating whether matrix A should be transposed before multiplication.
     * @param trans_b Flag indicating whether matrix B should be transposed before multiplication.
     * @param alpha Scaling factor for the matrix multiplication result.
     * @param beta Scaling factor for the output tensor (if it exists).
     * @param batch_count The number of matrices in the batch for batched matrix multiplication.
     */
    void InitMatMul(size_t m, size_t n, size_t k,
                    bool trans_a, bool trans_b,
                    float alpha, float beta,
                    size_t batch_count) override;

    /**
     * @brief Performs the forward pass of the MatMul layer using NVIDIA GPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `MatMul` 
     * class, providing a NVIDIA GPU-specific implementation for the matrix multiplication operation.
     */
    void Forward() override;

  private:
    cublasLtHandle_t handle_;              // cuBLASLt handle for GPU operations
    cublasLtMatmulDesc_t op_desc_;         // Matrix multiplication descriptor
    cublasLtMatrixLayout_t a_desc_, b_desc_, c_desc_; // Matrix layout descriptors for A, B, and C
    cublasLtMatmulAlgo_t algo_;            // Selected matmul algorithm
    void* work_space_;                     // Workspace memory for cuBLASLt operations
    size_t size_work_space_;               // Size of workspace memory

    void CUBLASLtSetup();
};

} // namespace oris_ai
