/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class Concat
 * @brief Represents a concatenation layer in a neural network.
 * 
 * This class defines a concatenation layer, which concatenates multiple input tensors 
 * along a specified dimension.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Concat : public LayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a concatenation layer.
     * @param name The name of the layer.
     */
    Concat(const std::string& layer_name) : LayerAbstract<T>(layer_name), concat_dim_(1) {}

    /**
     * @brief Virtual destructor for the Concat class.
     */
    virtual ~Concat() {}

    /**
     * @brief Initializes the concat layer with parameters from a TorchConcat object.
     * 
     * @param concat_params The TorchConcat object containing concat parameters.
     * @param target_device The device on which to perform the concatenation (e.g., CPU, GPU).
     */
    void InitConcat(const TorchConcat& concat_params, Device target_device);

    /**
     * @brief Pure virtual function to perform the forward pass of the concat layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the concat operation.
     */
    virtual void Forward() = 0;

  protected:
    size_t concat_dim_; // The dimension along which concatenation occurs
    std::vector<size_t> output_shape_; // Shape of the output tensor after concatenation
};

} // namespace oris_ai
