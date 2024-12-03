/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/maxpooling.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"

namespace oris_ai {

/**
 * @class MaxPoolingCPU
 * @brief A class that implements max pooling operations on a NVIDIA GPU.
 * 
 * This class provides a NVIDIA GPU-specific implementation for performing max pooling
 * operations, inheriting from the base `MaxPooling` class. It overrides the 
 * `Forward` method to execute the forward pass of the max pooling layer 
 * using NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class MaxPoolingGPU : public MaxPooling<T> {
  public:
    /**
     * @brief Constructor to initialize a MaxPoolingGPU layer without layer_name.
     */
    MaxPoolingGPU() : MaxPooling<T>() {}

    /**
     * @brief Constructor to initialize a MaxPoolingGPU layer with layer_name.
     */
    MaxPoolingGPU(const std::string& layer_name) : MaxPooling<T>(layer_name) {}

    /**
     * @brief Destructor for the MaxPoolingCPU class.
     */
    ~MaxPoolingGPU();

    /**
     * @brief Overrides the virtual InitMaxPooling function for NVIDIA GPU-specific
     * initialization.
     * 
     * This function implements the virtual InitMaxPooling method defined in the pooling
     * base class, configuring the max pooling layer with the provided parameters for
     * efficient execution on the NVIDIA GPU.
     * 
     * @param maxpool2d_params The TorchMaxPool2d object containing max pooling parameters.
     */
    void InitMaxPooling(const TorchMaxPool2d& maxpool2d_params) override;

    /**
     * @brief Performs the forward pass of the MaxPoolong layer using NVIDIA GPU.
     * 
     * This function overrides the pure virtual `Forward` method from the base `MaxPoolong` 
     * class, providing a NVIDIA GPU-specific implementation for the MaxPoolong operation.
     */
    void Forward() override;

  private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnPoolingDescriptor_t pool_desc_;
};

} // namespace oris_ai
