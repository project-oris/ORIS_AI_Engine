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
      Convolution() : activation_type_(ActivationType::NONE) {}

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
       * @brief Gets the weight tensor for the convolution layer.
       * 
       * @return A pointer to the weight tensor.
       */
      inline Tensor<T>* GetWeigt() { return weight_.get(); }

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

    protected:
      ActivationType activation_type_;           /** The type of activation function used in the layer */
      std::unique_ptr<Tensor<T>> weight_;        /** A unique pointer to the weight tensor */
      std::unique_ptr<Tensor<T>> bias_;          /** A unique pointer to the bias tensor, if applicable */
      std::unique_ptr<Tensor<T>> im2col_buffer_; /** A unique pointer to the buffer used for im2col operations */
  };

} // namespace oris_ai
