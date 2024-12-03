/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/upsample.h"

namespace oris_ai {

/**
 * @class UpsampleCPU
 * @brief Implements the forward pass of an upsampling layer on the CPU.
 * 
 * This class inherits from Upsample and implements the forward pass
 * specifically for CPU execution.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class UpsampleCPU : public Upsample<T> {
  public:
    /**
     * @brief Constructor to initialize a UpsampleCPU layer without layer_name.
     */
    UpsampleCPU() : Upsample<T>() {}

    /**
     * @brief Constructor to initialize the UpsampleCPU layer with layer_name.
     * @param name The name of the layer.
     */
    UpsampleCPU(const std::string& layer_name) : Upsample<T>(layer_name) {}

    /**
     * @brief Destructor for the UpsampleCPU class.
     */
    ~UpsampleCPU() {}

    /**
     * @brief Performs the forward pass for the upsample operation on CPU.
     */
    void Forward() override;
};

} // namespace oris_ai
