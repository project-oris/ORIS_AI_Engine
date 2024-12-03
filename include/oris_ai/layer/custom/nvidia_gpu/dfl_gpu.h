/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/custom/dfl.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"

namespace oris_ai {

/**
 * @class DFLGPU
 * @brief Represents a CPU-specific implementation of the Deep Feature Loss (DFL) layer.
 * 
 * This class defines the CPU-specific version of the DFL layer, inheriting from
 * the base DFL class and implementing the forward pass for NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DFLGPU : public DFL<T> {
  public:
    /**
     * @brief Constructor to initialize a DFLGPU layer.
     */
    DFLGPU() : DFL<T>(Device::GPU) {}

    /**
     * @brief Destructor for the DFLGPU class.
     */
    ~DFLGPU();

    /**
     * @brief Overrides the virtual InitDFL function for NVIDIA GPU-specific initialization.
     * 
     * This function implements the virtual InitDFL method defined in the DFL base class,
     * configuring the convolution layer with the provided parameters for efficient execution
     * on the NVIDIA GPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void InitDFL(const TorchConv2d& conv2d_params) override;

    /**
     * @brief Performs the forward pass of the DFL layer on the NVIDIA GPU.
     * 
     * This function implements the forward pass for the DFL layer using NVIDIA GPU operations.
     */
    void Forward() override;

    void DFLSoftmax(Tensor<T>& tensor) override;

  private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t tensor_desc_;
};

} // namespace oris_ai