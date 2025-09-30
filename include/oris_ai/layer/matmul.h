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

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class MatMul
 * @brief Represents a matrix multiplication layer in a neural network.
 * 
 * This class defines a matrix multiplication layer that performs matrix operations
 * between two input tensors with optional transposition and batch processing support.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class MatMul : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a matrix multiplication layer.
     * @param layer_name The name of the layer.
     */
    MatMul(const std::string& layer_name)
      : HiddenLayerAbstract<T>(layer_name),
        m_(0), n_(0), k_(0),
        trans_a_(false), trans_b_(false),
        alpha_(1.0f), beta_(0.0f),
        batch_count_(1),
        stride_a_(0), stride_b_(0), stride_c_(0) {}

    /**
     * @brief Virtual destructor for the MatMul class.
     */
    virtual ~MatMul() = default;

    /**
     * @brief Initializes the matrix multiplication layer with the specified parameters.
     * 
     * This function configures the matrix multiplication layer with dimensions,
     * transposition flags, scaling factors, and batch processing parameters.
     * 
     * @param m The number of rows in matrix A.
     * @param n The number of columns in matrix B.
     * @param k The number of columns in matrix A (and rows in matrix B).
     * @param trans_a Flag indicating whether to transpose matrix A.
     * @param trans_b Flag indicating whether to transpose matrix B.
     * @param alpha Scaling factor for the matrix multiplication result.
     * @param beta Scaling factor for the output tensor.
     * @param batch_count Number of matrices in the batch.
     */
    virtual void InitMatMul(size_t m, size_t n, size_t k,
                            bool trans_a, bool trans_b,
                            float alpha, float beta,
                            size_t batch_count) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass of the matrix multiplication layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the matrix multiplication.
     */
    virtual void Forward() = 0;

  protected:
    size_t m_, n_, k_;                    // Dimensions of the matrices
    bool trans_a_, trans_b_;              // Transposition flags for matrices A and B
    float alpha_, beta_;                  // Scaling factors for result and output
    size_t batch_count_;                  // Number of matrices in the batch
    size_t stride_a_, stride_b_, stride_c_; // Strides between matrices in batch

    void MatMulSetup(size_t m, size_t n, size_t k,
                    bool trans_a, bool trans_b,
                    float alpha, float beta,
                    size_t batch_count);
};

} // namespace oris_ai
