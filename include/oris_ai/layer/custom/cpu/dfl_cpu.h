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
    DFLCPU() : DFL<T>(Device::CPU) {}

    /**
     * @brief Default destructor for the DFLCPU class.
     */
    ~DFLCPU() = default;

    /**
     * @brief Overrides the virtual InitDFL function for CPU-specific initialization.
     * 
     * This function implements the virtual InitDFL method defined in the DFL base class,
     * configuring the convolution layer with the provided parameters for efficient execution
     * on the CPU.
     * 
     * @param conv2d_params The TorchConv2d object containing convolution parameters.
     */
    void InitDFL(const TorchConv2d& conv2d_params) override;

    /**
     * @brief Performs the forward pass of the DFL layer on the CPU.
     * 
     * This function implements the forward pass for the DFL layer using CPU operations.
     */
    void Forward() override;

    void DFLSoftmax(Tensor<T>& tensor) override;
};

} // namespace oris_ai
