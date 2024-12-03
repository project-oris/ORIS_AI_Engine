/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"

namespace oris_ai {

/**
 * @class ConcatGPU
 * @brief Represents a NVIDIA GPU-based concatenation layer in a neural network.
 * 
 * This class defines a concatenation layer that operates on NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class ConcatGPU : public Concat<T> {
  public:
    /**
     * @brief Constructor to initialize a ConcatGPU layer without layer_name.
     * @param name The name of the layer.
     */
    ConcatGPU() : Concat<T>() {}

    /**
     * @brief Constructor to initialize a ConcatGPU layer with layer_name.
     * @param name The name of the layer.
     */
    ConcatGPU(const std::string& layer_name) : Concat<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the ConcatGPU class.
     */
    ~ConcatGPU() = default;

    /**
     * @brief Performs the forward pass of the Concat layer using NVIDIA GPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Concat` 
     * class, providing a NVIDIA GPU-specific implementation for the Concat operation.
     */
    void Forward() override;
};

} // namespace oris_ai
