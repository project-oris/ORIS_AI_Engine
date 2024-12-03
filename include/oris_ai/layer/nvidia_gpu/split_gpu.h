/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/split.h"

namespace oris_ai {

/**
 * @class SplitGPU
 * @brief Represents a NVIDIA GPU-based Split layer in a neural network.
 * 
 * This class defines a Split layer that operates on the NVIDIA GPU, implementing
 * the transposition of specified dimensions in the input tensor.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class SplitGPU : public Split<T> {
  public:
    /**
     * @brief Default constructor to initialize a SplitGPU layer without layer_name.
     */
    SplitGPU() : Split<T>() {}

    /**
     * @brief Virtual destructor for the SplitGPU class.
     */
    ~SplitGPU() = default;

    /**
     * @brief Performs the forward pass of the Split operation using NVIDIA GPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Split` 
     * class, providing a NVIDIA GPU-specific implementation for the Split operation.
     */
    void Forward() override;
};

} // namespace oris_ai
