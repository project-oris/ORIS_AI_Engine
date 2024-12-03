/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/maxpooling.h"

namespace oris_ai {

/**
 * @class MaxPoolingCPU
 * @brief A class that implements max pooling operations on a CPU.
 * 
 * This class provides a CPU-specific implementation for performing max pooling
 * operations, inheriting from the base `MaxPooling` class. It overrides the 
 * `Forward` method to execute the forward pass of the max pooling layer 
 * using CPU resources.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
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
     * @brief Overrides the virtual InitMaxPooling function for CPU-specific initialization.
     * 
     * This function implements the virtual InitMaxPooling method defined in the pooling
     * base class, configuring the max pooling layer with the provided parameters for
     * efficient execution on the CPU.
     * 
     * @param maxpool2d_params The TorchMaxPool2d object containing max pooling parameters.
     */
    void InitMaxPooling(const TorchMaxPool2d& maxpool2d_params) override;

    /**
     * @brief Performs the forward pass of the MaxPoolong layer using CPU resources.
     * 
     * This function overrides the pure virtual `Forward` method from the base `MaxPoolong` 
     * class, providing a CPU-specific implementation for the MaxPoolong operation.
     */
    void Forward() override;
};

} // namespace oris_ai
