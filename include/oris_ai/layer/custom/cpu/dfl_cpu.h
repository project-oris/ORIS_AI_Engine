/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/custom/dfl.h"

namespace oris_ai {

/**
 * @class DFLCPU
 * @brief Represents a CPU-specific implementation of the Deep Feature Loss (DFL) layer.
 * 
 * This class defines the CPU-specific version of the DFL layer, inheriting from
 * the base DFL class and implementing the forward pass for CPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DFLCPU : public DFL<T> {
  public:
    /**
     * @brief Constructor to initialize a DFLCPU layer.
     */
    DFLCPU() : DFL<T>() {}

    /**
     * @brief Default destructor for the DFLCPU class.
     */
    ~DFLCPU() = default;

    /**
     * @brief Performs the forward pass of the DFL layer on the CPU.
     * 
     * This function implements the forward pass for the DFL layer using CPU operations.
     */
    void Forward() override;

    void DFLSoftmax(Tensor<T>& tensor) override;
};

} // namespace oris_ai
