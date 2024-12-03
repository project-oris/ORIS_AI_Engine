/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class Transpose
 * @brief Represents a transpose layer in a neural network.
 * 
 * This class defines a transpose layer that performs transposition
 * by swapping two specified dimensions of the input tensor.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Transpose : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Default constructor to initialize a transpose layer without layer_name.
     */
    Transpose() : HiddenLayerAbstract<T>() {}

    /**
     * @brief Constructor to initialize a transpose layer with a specified layer_name.
     * @param layer_name The name of the layer.
     */
    // Transpose(const std::string& layer_name) : HiddenLayerAbstract<T>(layer_name) {}

    /**
     * @brief Destructor for the Transpose class.
     */
    ~Transpose() = default;

    /**
     * @brief Initializes the transpose layer by setting dimensions to swap and optional view shape usage.
     * 
     * This virtual function finalizes the transpose layer setup by applying device-specific 
     * configurations for the layer on CPU, GPU, or other hardware. It uses the initial
     * parameters prepared by TransposeSetup() and extends the setup to include any
     * device-specific optimizations or settings.
     * 
     * @param dim1 The first dimension to swap.
     * @param dim2 The second dimension to swap.
     * @param use_view_input_shape Flag indicating whether to use the view shape of the input
     * tensor.
     */
    virtual void InitTranspose(size_t dim1, size_t dim2, bool use_view_input_shape = false) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass of the transpose layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the transposition.
     */
    virtual void Forward() = 0;
  
  protected:
    /**
     * @brief Initializes the transpose layer by setting dimensions to swap and optional view
     * shape usage.
     * 
     * This function configures the transpose layer to swap the specified dimensions
     * and prepares internal structures, such as output shape and strides.
     * 
     * @param dim1 The first dimension to swap.
     * @param dim2 The second dimension to swap.
     * @param use_view_input_shape Flag indicating whether to use the view shape of the input
     * tensor.
     */
    void TransposeSetup(size_t dim1, size_t dim2, bool use_view_input_shape = false);

    std::vector<size_t> transposed_index_;  // Precomputed indices for transposed tensor
};

} // namespace oris_ai
