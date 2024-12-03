/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/transpose.h"

namespace oris_ai {

/**
 * @class TransposeGPU
 * @brief Represents a NVIDIA GPU-based transpose layer in a neural network.
 * 
 * This class defines a transpose layer that operates on the NVIDIA GPU, implementing
 * the transposition of specified dimensions in the input tensor.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class TransposeGPU : public Transpose<T> {
  public:
    /**
     * @brief Default constructor to initialize a TransposeGPU layer without a specified layer_name.
     */
    TransposeGPU() : Transpose<T>() {}

    /**
     * @brief Destructor for the TransposeGPU class.
     */
    ~TransposeGPU();

    /**
     * @brief Overrides the virtual InitTranspose function for NVIDIA GPU-specific
     * initialization.
     * 
     * This function implements the virtual InitTranspose method defined in the Transpose 
     * base class, configuring the transpose layer with the provided parameters for
     * efficient execution on the NVIDIA GPU.
     * 
     * @param dim1 The first dimension to swap.
     * @param dim2 The second dimension to swap.
     * @param use_view_input_shape Flag indicating whether to use the view shape of the input
     * tensor.
     */
    void InitTranspose(size_t dim1, size_t dim2, bool use_view_input_shape = false) override;

    /**
     * @brief Performs the forward pass of the transpose operation using NVIDIA GPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Transpose` 
     * class, providing a NVIDIA GPU-specific implementation for the transpose operation.
     * It rearranges data from the input tensor to the output tensor based on the precomputed
     * transposed indices, which are set in the `InitTranspose` function.
     */
    void Forward() override;

  private:
    size_t* cuda_transposed_index_;
};

} // namespace oris_ai
