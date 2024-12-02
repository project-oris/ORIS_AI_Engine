/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/permute.h"

namespace oris_ai {

/**
 * @class PermuteCPU
 * @brief Permute layer implementation for CPU.
 * 
 * This class performs the permute operation for tensors on the CPU. 
 * It inherits from the Permute class, which provides the core functionality for 
 * handling the permute operation.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class PermuteCPU : public Permute<T> {
 public:
  /**
   * @brief Constructor for PermuteCPU.
   * @param layer_name The name of the layer.
   */
  PermuteCPU(const std::string& layer_name) : Permute<T>(layer_name) {}

  /**
   * @brief Destructor for PermuteCPU.
   */
  ~PermuteCPU() = default;

  /**
   * @brief Forward pass for the PermuteCPU layer.
   * 
   * This function overrides the pure virtual `Forward` method from the base `Permute` 
   * class, providing a CPU-specific implementation for the Permute operation.
   */
  void Forward() override;
};

}  // namespace oris_ai
