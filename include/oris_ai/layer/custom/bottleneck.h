/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/convolution.h"

namespace oris_ai {

/**
 * @class Bottleneck
 * @brief Represents a bottleneck layer in YOLO model.
 * 
 * This class encapsulates two convolutional layers, applies element-wise addition
 * if the input and output channels are the same, and manages forward pass and tensor 
 * operations.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Bottleneck : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a Bottleneck layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU).
     * @param add Flag to determine if element-wise addition is performed.
     */
    Bottleneck(const std::string& layer_name, Device target_device, bool add);

    /**
     * @brief Destructor for the Bottleneck class.
     */
    ~Bottleneck() = default;

    /**
     * @brief Initializes the Bottleneck layer with two convolutional layers.
     * 
     * @param layer_conv1 The TorchLayer object for the first convolution layer.
     * @param layer_act1 The TorchLayer object containing the activation type.
     * @param layer_conv2 The TorchLayer object for the second convolution layer.
     * @param layer_act2 The TorchLayer object containing the activation type.
     */
    void InitBottleneck(const TorchLayer& layer_conv1,  const TorchLayer& layer_act1, const TorchLayer& layer_conv2, const TorchLayer& layer_act2);

    const std::vector<TorchLayer> c2f_layers;

    /**
     * @brief Sets the input tensor for the first convolutional layer in the bottleneck.
     * @param input_tensor The input tensor.
     */
    void SetInputTensor(Tensor<T>* input_tensor) override;

    /**
     * @brief Gets the number of input tensors.
     * 
     * @return The number of input tensors.
     */
    inline size_t GetInputSize() override { return bottleneck_cv1_->GetInputSize(); }

    /**
     * @brief Retrieves the output tensor from the second convolutional layer in the bottleneck.
     * @return The output tensor from the second convolutional layer.
     */
    Tensor<T>* GetOutputTensor() override;

    /**
     * @brief Perform the forward pass of the Bottleneck layer.
     */
    void Forward() override;

  private:
    std::unique_ptr<Convolution<T>> bottleneck_cv1_;  // First convolutional layer in bottleneck
    std::unique_ptr<Convolution<T>> bottleneck_cv2_;  // Second convolutional layer in bottleneck
    bool add_;  // Flag to determine if element-wise addition is performed
};

/**
 * @brief Factory function to create a Bottleneck layer.
 * 
 * This function dynamically creates a Bottleneck layer based on target device.
 * 
 * @tparam T The data type used by the Bottleneck layer (e.g., float).
 * @param layer_name The name of the Bottleneck layer.
 * @param target_device The device (CPU or GPU) on which the Bottleneck layer will operate.
 * @param add Flag to determine if element-wise addition is performed.
 */
template <typename T>
std::unique_ptr<Bottleneck<T>> CreateBottleneck(const std::string& layer_name, Device target_device, bool add);

} // namespace oris_ai
