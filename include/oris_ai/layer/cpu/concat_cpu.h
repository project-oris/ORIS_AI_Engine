/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"

namespace oris_ai {

/**
 * @class ConcatCPU
 * @brief Represents a CPU-based concatenation layer in a neural network.
 * 
 * This class defines a concatenation layer that operates on CPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConcatCPU : public Concat<T> {
  public:
    /**
     * @brief Constructor to initialize a ConcatCPU layer without layer_name.
     * @param name The name of the layer.
     */
    ConcatCPU() : Concat<T>() {}

    /**
     * @brief Constructor to initialize a ConcatCPU layer with layer_name.
     * @param name The name of the layer.
     */
    ConcatCPU(const std::string& layer_name) : Concat<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the ConcatCPU class.
     */
    ~ConcatCPU() = default;

    /**
     * @brief Performs the forward pass of the Concat layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Concat` 
     * class, providing a CPU-specific implementation for the Concat operation.
     */
    void Forward() override;
};

} // namespace oris_ai
