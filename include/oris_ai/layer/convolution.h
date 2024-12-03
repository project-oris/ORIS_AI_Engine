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
class Convolution : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a Convolution layer without layer_name.
     */
    Convolution()
      : HiddenLayerAbstract<T>(),
        activation_type_(ActivationType::NONE), 
        is_1x1_conv_(false) {}
        
    /**
     * @brief Constructor to initialize a Convolution layer with layer_name.
     * @param name The name of the layer.
     */
    Convolution(const std::string& layer_name) 
      : HiddenLayerAbstract<T>(layer_name), 
        activation_type_(ActivationType::NONE), 
        is_1x1_conv_(false) {}

    /**
     * @brief Destructor for the Convolution class.
     */
    ~Convolution() = default;

    /**
     * @brief Initializes the convolution layer on the specific device with parameters from a
     * TorchConv2d object.
     * 
     * This virtual function finalizes the convolution layer setup by applying device-specific 
     * configurations for the layer on CPU, GPU, or other hardware. It uses the initial
     * parameters prepared by ConvolutionSetup() and extends the setup to include any
     * device-specific optimizations or settings.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    virtual void InitConvolution(const TorchConv2d& conv2d_params) = 0;

    /**
     * @brief Sets the activation function for the convolution layer.
     * 
     * This function defines the activation function to be applied after the convolution operation.It sets the activation type (e.g., ReLU, SiLU) and related parameters for the layer.
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
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the convoltion operation.
     */
    virtual void Forward() = 0;

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints the weight tensor for debugging purposes.
     * 
     * This function is available only in debug mode and is used to log the current values
     * in the weight tensor for the layer.
     */
    void PrintWeight();
#endif

  protected:
    /**
     * @brief Configures the convolutional layer with the given parameters from a TorchConv2d
     * object.
     * 
     * This function performs the initial setup of convolution layer parameters, such as
     * configuring the weight tensor, bias (if applicable), kernel size, stride, padding, and
     * output dimensions. It does not handle device-specific implementation details, allowing
     * InitConvolution() to  manage these aspects for specific devices like CPU or GPU through
     * virtual function overrides.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void ConvolutionSetup(const TorchConv2d& conv2d_params);

    ActivationType activation_type_;    // The type of activation function used in the layer.
    std::unique_ptr<Tensor<T>> weight_; // A unique pointer to the weight tensor for the layer.
    std::unique_ptr<Tensor<T>> bias_;   // A unique pointer to the bias tensor, if applicable.
    std::unique_ptr<Tensor<T>> im2col_buffer_; // A unique pointer to the buffer used for im2col operations.

    size_t out_channels_, in_channels_;   // The number of output and input channels for the layer.
    size_t kernel_h_, kernel_w_;          // The height and width of the convolution kernel.
    size_t stride_h_, stride_w_;          // The stride along the height and width dimensions.
    size_t padding_h_, padding_w_;        // The padding size for the height and width dimensions.
    size_t output_height_, output_width_; // The height and width of the output tensor.

    bool is_1x1_conv_;  // Flag indicating whether this is a 1x1 convolution (special optimization case).
};

} // namespace oris_ai
