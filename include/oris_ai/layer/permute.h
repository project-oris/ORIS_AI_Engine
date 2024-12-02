/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class Permute
 * @brief Permute layer for rearranging dimensions in neural networks.
 * 
 * This class represents a permute layer that rearranges the dimensions of the input tensor.
 * It inherits from HiddenLayerAbstract and allows flexible dimension changes, including 
 * dimension reduction (e.g., from 4D to 3D).
 * 
 * @tparam T Data type for layer operations (e.g., float).
 */
template <typename T>
class Permute : public HiddenLayerAbstract<T> {
public:
  /**
   * @brief Constructor to initialize a Permute layer without layer name.
   */
  Permute() : HiddenLayerAbstract<T>() {}

  /**
   * @brief Constructor to initialize a Permute layer with a given name.
   * 
   * @param layer_name The name of the Permute layer.
   */
  Permute(const std::string& layer_name) : HiddenLayerAbstract<T>(layer_name) {}

  /**
   * @brief Destructor for the Permute class.
   */
  ~Permute() = default;

  /**
   * @brief Initialize the Permute layer with target shape.
   * 
   * @param target_shape A vector specifying the target shape after permutation.
   */
  void InitPermute(const std::vector<size_t>& target_shape);

  /**
   * @brief Virtual function to perform the forward pass (to be implemented by derived classes).
   */
  virtual void Forward() = 0;

protected:
  std::vector<size_t> target_shape_;   // Target shape after permutation.
  // void ComputeFinalShape();            // Compute the final shape by resolving -1 dimensions.
};

} // namespace oris_ai
