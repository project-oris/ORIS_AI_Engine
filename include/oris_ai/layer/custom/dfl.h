/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"
#include "oris_ai/layer/transpose.h"

namespace oris_ai {

/**
 * @class DFL
 * @brief Represents a specialized Deep Feature Loss (DFL) layer.
 * 
 * This class defines a DFL layer, which is a simple 1x1 convolution without bias
 * or activation. The DFL layer is intended for lightweight transformations.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DFL : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Default constructor for the DFL layer without layer_name.
     */
    DFL(Device target_device);

    /**
     * @brief Default destructor for the DFL class.
     */
    ~DFL() = default;

    /**
     * @brief Initializes the DFL layer with parameters from a TorchConv2d object.
     * 
     * This virtual function finalizes the DFL layer setup by applying device-specific     * 
     * configurations for the layer on CPU, GPU, or other hardware. It uses the initial
     * parameters prepared by DFLSetup() and extends the setup to include any
     * device-specific optimizations or settings.
     * 
     * @param conv2d_params The TorchConv2d object containing the convolution parameters.
     */
    virtual void InitDFL(const TorchConv2d& conv2d_params) = 0;

    /**
     * @brief Pure virtual function to perform the forward pass of the DFL layer.
     * 
     * This function must be implemented by derived classes, such as DFLCPU or DFLGPU,
     * which are specific to the execution device (e.g., CPU or GPU). 
     */
    virtual void Forward() = 0;

    /**
     * @brief Pure virtual function to apply the softmax operation for the DFL layer.
     * 
     * This function must be implemented by derived classes, such as DFLCPU or DFLGPU,
     * which are specific to the execution device (e.g., CPU or GPU).
     * The softmax operation is applied to the provided tensor to normalize values.
     * 
     * @param tensor A reference to the tensor on which to apply the softmax operation.
     */
    virtual void DFLSoftmax(Tensor<T>& tensor) = 0;

  protected:
    /**
     * @brief Initializes the DFL layer with parameters from a TorchConv2d object.
     * 
     * This function sets up the DFL layer by configuring the weight tensor without bias.
     * 
     * @param conv2d_params The TorchConv2d object containing the convolution parameters.
     */
    void DFLSetup(const TorchConv2d& conv2d_params);

    std::unique_ptr<Tensor<T>> weight_; // Unique pointer to the weight tensor for the layer
    size_t in_channels_;                // Number of input channels
    size_t out_channels_;               // Number of output channels (1x1 convolution)
    size_t output_height_;              // Height of the output tensor
    size_t output_width_;               // Width of the output tensor

    std::unique_ptr<Transpose<T>> trans_boxes_;
};

} // namespace oris_ai
