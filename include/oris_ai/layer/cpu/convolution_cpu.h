/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/convolution.h"

namespace oris_ai {

/**
 * @class ConvolutionCPU
 * @brief A class that implements convolution operations on a CPU.
 * 
 * This class provides a CPU-specific implementation for performing convolution 
 * operations, inheriting from the base `Convolution` class. It overrides the 
 * `Forward` method to perform the forward pass of the convolution layer 
 * using CPU resources.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConvolutionCPU : public Convolution<T> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionCPU layer.
     * @param name The name of the layer.
     */
    ConvolutionCPU(const std::string& name) : Convolution<T>(name) {}

    /**
     * @brief Destructor for the ConvolutionCPU class.
     */
    ~ConvolutionCPU() {}

    /**
     * @brief Performs the forward pass of the convolution layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Convolution` 
     * class, providing a CPU-specific implementation for the convolution operation.
     */
    void Forward() override;

  private:
    /**
     * @brief Performs the im2col operation for convolution.
     * 
     * The im2col (image to column) operation is used to transform the input data 
     * into a format suitable for matrix multiplication in the convolution process.
     * 
     * @param input_data Pointer to the input data.
     * @param col_data Pointer to the column data buffer.
     * @param input_h Height of the input data.
     * @param input_w Width of the input data.
     * @param kernel_h Height of the convolution kernel.
     * @param kernel_w Width of the convolution kernel.
     * @param pad_h Padding size along the height.
     * @param pad_w Padding size along the width.
     * @param stride_h Stride size along the height.
     * @param stride_w Stride size along the width.
     */
    void Im2Col(const T* input_data, T* col_data, int input_h, int input_w,
                int kernel_h, int kernel_w, int pad_h, int pad_w,
                int stride_h, int stride_w);
};

} // namespace oris_ai
