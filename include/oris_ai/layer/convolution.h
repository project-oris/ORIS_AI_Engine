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
   * @tparam T The data type for the layer operations (e.g., float).
   */
  template <typename T>
  class Convolution : public LayerAbstract<T> {
    public:
      /**
       * @brief Constructor to initialize a Convolution layer with no activation.
       */
      Convolution(const std::string& name) : LayerAbstract<T>(name), activation_type_(ActivationType::NONE), is_1x1_conv_(false) {}

      /**
       * @brief Virtual destructor for the Convolution class.
       */
      virtual ~Convolution() {}

      /**
       * @brief Initializes the convolution layer with parameters.
       * 
       * @param conv2d The TorchConv2d object containing convolution parameters.
       * @param target_device The device on which to perform the convolution (e.g., CPU, GPU).
       */
      void InitConvolution(const TorchConv2d& conv2d, Device target_device);

      /**
       * @brief Sets the activation function for the convolution layer.
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
        if (bias_)
          return bias_.get();
        else
          return nullptr;
      }

      /**
       * @brief Pure virtual function to perform the forward pass of the convolution layer.
       */
      virtual void Forward() = 0;

      /**
       * @brief Prints weight tensor (for debug).
       */
      void PrintWeight();

    protected:
      ActivationType activation_type_;    // The type of activation function used in the layer
      std::unique_ptr<Tensor<T>> weight_; // A unique pointer to the weight tensor
      std::unique_ptr<Tensor<T>> bias_;   // A unique pointer to the bias tensor, if applicable
      std::unique_ptr<Tensor<T>> im2col_buffer_; // A unique pointer to the buffer used for im2col operations

      size_t out_channels_, in_channels_;   // Channel information
      size_t kernel_h_, kernel_w_;          // Kernel dimensions
      size_t stride_h_, stride_w_;          // Stride dimensions
      size_t padding_h_, padding_w_;        // Padding dimensions      
      size_t output_height_, output_width_; // Output dimensions

      bool is_1x1_conv_;  // Determine whether this conv is a 1x1 conv
  };

} // namespace oris_ai
