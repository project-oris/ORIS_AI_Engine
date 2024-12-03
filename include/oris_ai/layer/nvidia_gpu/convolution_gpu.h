/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/convolution.h"
#include "oris_ai/accelerator/cudnn/cudnn_manager.h"

namespace oris_ai {

/**
 * @class ConvolutionGPU
 * @brief A class that implements convolution operations on a NVIDIA GPU.
 * 
 * This class provides a NVIDIA GPU-specific implementation for performing convolution 
 * operations, inheriting from the base `Convolution` class. It overrides the 
 * `Forward` method to perform the forward pass of the convolution layer 
 * using NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConvolutionGPU : public Convolution<T> {
  public:
    /**
     * @brief Constructor to initialize a ConvolutionGPU layer without layer_name.
     */
    ConvolutionGPU() : Convolution<T>(), work_space_(nullptr), size_work_space_(0) {}

    /**
     * @brief Constructor to initialize a ConvolutionGPU layer with layer_name.
     * @param name The name of the layer.
     */
    ConvolutionGPU(const std::string& layer_name) : Convolution<T>(layer_name), work_space_(nullptr), size_work_space_(0) {}

    /**
     * @brief Destructor for the ConvolutionGPU class.
     */
    ~ConvolutionGPU();

    /**
     * @brief Overrides the virtual InitConvolution function for NVIDIA GPU-specific
     * initialization.
     * 
     * This function implements the virtual InitConvolution method defined in the Convolution 
     * base class, configuring the convolution layer with the provided parameters for
     * efficient execution on the NVIDIA GPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void InitConvolution(const TorchConv2d& conv2d_params) override;

    /**
     * @brief Performs the forward pass of the Convolution layer using NVIDIA GPU.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Convolution` 
     * class, providing a NVIDIA GPU-specific implementation for the convolution operation.
     */
    void Forward() override;

  private:
    cudnnHandle_t cudnn_handle_;
    cudnnTensorDescriptor_t input_desc_, output_desc_;
    cudnnFilterDescriptor_t filter_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;
    cudnnTensorDescriptor_t bias_desc_;

    // activation
    // cudnnActivationDescriptor_t act_desc_;

    // cudnn convolution forward algorithm
    cudnnConvolutionFwdAlgo_t conv_algo_;

    void* work_space_;
    size_t size_work_space_;
};

} // namespace oris_ai
