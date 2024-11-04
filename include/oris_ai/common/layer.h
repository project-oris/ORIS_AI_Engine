/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include <string>  // Include for std::string

namespace oris_ai {

  /**
   * @enum LayerType
   * @brief Defines types of layers that can be used in the neural network.
   */
  enum class LayerType {
    CONV,
    BATCHNORM,
    ACTIVATION,
    MAXPOOL,
    CONCAT,
    UPSAMPLE   
  };

  /**
   * @class Layer
   * @brief Represents a neural network layer with a specified name.
   * 
   * @tparam T The data type for the layer operations (e.g., float, int).
   */
  template <typename T>
  class Layer {
  public:
    /**
     * @brief Constructor to create a Layer object with a given name.
     * 
     * @param layer_name The name of the layer.
     */
    Layer(const std::string& layer_name);

  private:
    std::string layer_name_;  /* The name of the layer */
  };

} // namespace oris_ai
