/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class MaxPooling
 * @brief Represents a max pooling layer in a neural network.
 * 
 * This class defines a max pooling layer.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class MaxPooling : public LayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a max pooling layer.
     * @param name The name of the layer.
     */
    MaxPooling(const std::string& name) : LayerAbstract<T>(name) {}

    /**
     * @brief Virtual destructor for the MaxPooling class.
     */
    virtual ~MaxPooling() {}

    /**
     * @brief Initializes the pooling layer with parameters from a TorchMaxPool2d object.
     * 
     * @param maxpool2d The TorchMaxPool2d object.
     * @param target_device The device on which to perform the max pooling (e.g., CPU, GPU).
     */
    void InitMaxPooling(const TorchMaxPool2d& maxpool2d, Device target_device);

    /**
     * @brief Pure virtual function to perform the forward pass of the max pool layer.
     * 
     * This function should be implemented by the derived classes (e.g., CPU or GPU-specific versions).
     */
    virtual void Forward() = 0;

  protected:
    size_t kernel_size_;    // Kernel dimensions for the pooling operation
    size_t stride_;         // Stride dimensions for the pooling operation
    size_t padding_;        // Padding dimensions for the pooling operation
    size_t output_height_, output_width_; // Output dimensions after the pooling operation
    bool ceil_mode_;        // Flag to indicate if ceil mode is enabled
};

} // namespace oris_ai
