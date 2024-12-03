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
     * @brief Initializes the max pooling layer on the specific device with parameters from a
     * TorchMaxPool2d object.
     * 
     * This virtual function finalizes the max pooling layer setup by applying device-specific 
     * configurations for the layer on CPU, GPU, or other hardware. It uses the initial
     * parameters prepared by MaxPoolingSetup() and extends the setup to include any
     * device-specific optimizations or settings.
     * 
     * @param maxpool2d_params The TorchMaxPool2d object containing max pooling parameters.
     */
    virtual void InitMaxPooling(const TorchMaxPool2d& maxpool2d_params) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass of the max pooling layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the pooling operation.
     */
    virtual void Forward() = 0;

  protected:
    /**
     * @brief Configures the max pooling layer with the given parameters from a TorchMaxPool2d
     * object.
     * 
     * This function performs the initial setup of max pooling layer parameters, such as
     * configuring the kernel size, stride, padding, and output dimensions based on the
     * TorchMaxPool2d parameters. It does not handle device-specific implementation details,
     * allowing InitMaxPooling() to manage these aspects for specific devices like CPU or GPU
     * through virtual function overrides.
     * 
     * @param maxpool2d_params The TorchMaxPool2d object containing max pooling parameters.
     */
    void MaxPoolingSetup(const TorchMaxPool2d& maxpool2d_params);

    size_t kernel_size_;    // The size of the pooling kernel
    size_t stride_;         // The stride of the pooling operation
    size_t padding_;        // The padding size for the pooling operation
    size_t output_height_, output_width_; // The output dimensions after the pooling operation
    bool ceil_mode_;        // Flag to indicate if ceil mode is enabled (true) or not (false)
};

} // namespace oris_ai
