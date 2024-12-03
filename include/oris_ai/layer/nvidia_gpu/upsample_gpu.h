/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/upsample.h"

namespace oris_ai {

/**
 * @class UpsampleGPU
 * @brief Implements the forward pass of an upsampling layer on the NVIDIA GPU.
 * 
 * This class inherits from Upsample and implements the forward pass
 * specifically for NVIDIA GPU execution.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class UpsampleGPU : public Upsample<T> {
  public:
    /**
     * @brief Constructor to initialize a UpsampleGPU layer without layer_name.
     */
    UpsampleGPU() : Upsample<T>() {}

    /**
     * @brief Constructor to initialize the UpsampleGPU layer with layer_name.
     * @param name The name of the layer.
     */
    UpsampleGPU(const std::string& layer_name) : Upsample<T>(layer_name) {}

    /**
     * @brief Destructor for the UpsampleGPU class.
     */
    ~UpsampleGPU() {}

    /**
     * @brief Performs the forward pass for the upsample operation on NVIDIA GPU.
     */
    void Forward() override;
};

} // namespace oris_ai
