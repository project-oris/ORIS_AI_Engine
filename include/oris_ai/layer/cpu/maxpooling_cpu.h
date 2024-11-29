/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/maxpooling.h"

namespace oris_ai {

template <typename T>
class MaxPoolingCPU : public MaxPooling<T> {
  public:
    /**
     * @brief Constructor to initialize a MaxPoolingCPU layer without layer_name.
     */
    MaxPoolingCPU() : MaxPooling<T>() {}

    /**
     * @brief Constructor to initialize a MaxPoolingCPU layer with layer_name.
     */
    MaxPoolingCPU(const std::string& layer_name) : MaxPooling<T>(layer_name) {}

    /**
     * @brief Destructor for the MaxPoolingCPU class.
     */
    ~MaxPoolingCPU() = default;

    /**
     * @brief Performs the forward pass of the MaxPoolong layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `MaxPoolong` 
     * class, providing a CPU-specific implementation for the MaxPoolong operation.
     */
    void Forward() override;
};

} // namespace oris_ai
