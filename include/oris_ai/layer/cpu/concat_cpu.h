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
     * @brief Constructor to initialize a ConcatCPU layer.
     * @param name The name of the layer.
     */
    ConcatCPU(const std::string& name) : Concat<T>(name) {}

    /**
     * @brief Virtual destructor for the ConcatCPU class.
     */
    virtual ~ConcatCPU() {}

    /**
     * @brief Implements the forward pass for the concatenation layer.
     */
    void Forward() override;
};

} // namespace oris_ai
