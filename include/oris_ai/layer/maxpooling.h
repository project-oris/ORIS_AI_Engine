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
 * This class defines a max pooling layer that performs downsampling
 * by applying the max pooling operation over input feature maps.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class MaxPooling : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a max pooling layer without layer_name.
     * @param name The name of the layer.
     */
    MaxPooling() : HiddenLayerAbstract<T>() {}

    /**
     * @brief Constructor to initialize a max pooling layer with layer_name.
     * @param name The name of the layer.
     */
    MaxPooling(const std::string& layer_name) : HiddenLayerAbstract<T>(layer_name) {}

    /**
     * @brief Destructor for the MaxPooling class.
     */
    ~MaxPooling() = default;

    /**
     * @brief Initializes the pooling layer with parameters from a TorchMaxPool2d object.
     * 
     * @param maxpool2d_params The TorchMaxPool2d object containing max pooling parameters.
     */
    void InitMaxPooling(const TorchMaxPool2d& maxpool2d_params);

    /**
     * @brief Pure virtual function to perform the forward pass of the max pooling layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the pooling operation.
     */
    virtual void Forward() = 0;

  protected:
    size_t kernel_size_;    // The size of the pooling kernel
    size_t stride_;         // The stride of the pooling operation
    size_t padding_;        // The padding size for the pooling operation
    size_t output_height_, output_width_; // The output dimensions after the pooling operation
    bool ceil_mode_;        // Flag to indicate if ceil mode is enabled (true) or not (false)
};

} // namespace oris_ai
