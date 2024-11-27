/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class Convolution
 * @brief Represents a convolutional layer in a neural network.
 * 
 * This class defines a generic convolutional layer, including its parameters,
 * activation functions, and required buffers for operations like im2col.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Convolution : public LayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a Convolution layer with no activation.
     * @param name The name of the layer.
     */
    Convolution(const std::string& name) : LayerAbstract<T>(name), activation_type_(ActivationType::NONE), is_1x1_conv_(false) {}

    /**
     * @brief Virtual destructor for the Convolution class.
     */
    virtual ~Convolution() {}

    /**
     * @brief Initializes the convolution layer with parameters from a TorchConv2d object.
     * 
     * This function sets up the convolutional layer's weight, bias, and other necessary parameters
     * based on the provided TorchConv2d object, and configures it to run on the specified device.
     * 
     * @param conv2d The TorchConv2d object containing convolution parameters.
     * @param target_device The device on which to perform the convolution (e.g., CPU, GPU).
     */
    void InitConvolution(const TorchConv2d& conv2d, Device target_device);

    /**
     * @brief Sets the activation function for the convolution layer.
     * 
     * This function specifies the activation function to be applied after the convolution operation.
     * 
     * @param act The TorchActivation object containing the activation type and its parameters.
     */
    void SetActivation(const TorchActivation& act);

    /**
     * @brief Gets the weight tensor for the convolution layer.
     * 
     * @return A pointer to the weight tensor.
     */
    inline Tensor<T>* GetWeight() { return weight_.get(); }

    /**
     * @brief Gets the bias tensor for the convolution layer, if available.
     * 
     * @return A pointer to the bias tensor, or nullptr if no bias is present.
     */
    inline Tensor<T>* GetBias() {
      return bias_ ? bias_.get() : nullptr;
    }

    /**
     * @brief Pure virtual function to perform the forward pass of the convolution layer.
     * 
     * This function should be implemented by the derived classes (e.g., CPU or GPU-specific versions).
     */
    virtual void Forward() = 0;

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints the weight tensor for debugging purposes.
     */
    void PrintWeight();
#endif

  protected:
    ActivationType activation_type_;    // The type of activation function used in the layer.
    std::unique_ptr<Tensor<T>> weight_; // A unique pointer to the weight tensor.
    std::unique_ptr<Tensor<T>> bias_;   // A unique pointer to the bias tensor, if applicable.
    std::unique_ptr<Tensor<T>> im2col_buffer_; // A unique pointer to the buffer used for im2col operations.

    size_t out_channels_, in_channels_;   // Channel information (number of input/output channels).
    size_t kernel_h_, kernel_w_;          // Kernel dimensions (height and width).
    size_t stride_h_, stride_w_;          // Stride dimensions (height and width).
    size_t padding_h_, padding_w_;        // Padding dimensions (height and width).
    size_t output_height_, output_width_; // Output tensor dimensions (height and width).

    bool is_1x1_conv_;  ///< Flag to determine whether this is a 1x1 convolution.
};

} // namespace oris_ai
