/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/split.h"

namespace oris_ai {

/**
 * @class SplitCPU
 * @brief Represents a CPU-based Split layer in a neural network.
 * 
 * This class defines a Split layer that operates on the CPU, implementing
 * the transposition of specified dimensions in the input tensor.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class SplitCPU : public Split<T> {
  public:
    /**
     * @brief Default constructor to initialize a SplitCPU layer without a specified layer_name.
     */
    SplitCPU() : Split<T>() {}

    /**
     * @brief Constructor to initialize a SplitCPU layer with a specified layer_name.
     * @param layer_name The name of the layer.
     */
    // SplitCPU(const std::string& layer_name) : Split<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the SplitCPU class.
     */
    ~SplitCPU() = default;

    /**
     * @brief Performs the forward pass of the Split operation using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `Split` 
     * class, providing a CPU-specific implementation for the Split operation.
     */
    void Forward() override;
};

} // namespace oris_ai
