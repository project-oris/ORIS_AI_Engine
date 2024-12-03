/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/custom/decode_bboxes.h"

namespace oris_ai {

/**
 * @class DecodeBboxesGPU
 * @brief Represents a CPU-specific implementation of the DecodeBboxes layer.
 * 
 * This class defines the CPU-specific version of the DecodeBboxes layer, inheriting from
 * the base DecodeBboxes class and implementing the forward pass for NVIDIA GPU.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DecodeBboxesGPU : public DecodeBboxes<T> {
  public:
    /**
     * @brief Constructor to initialize a DecodeBboxesGPU layer with layer_name.
     * @param layer_name The name of the layer.
     */
    DecodeBboxesGPU(const std::string& layer_name) : DecodeBboxes<T>(layer_name, Device::GPU) {}

    /**
     * @brief Default destructor for the DecodeBboxesGPU class.
     */
    ~DecodeBboxesGPU() = default;

    /**
     * @brief Performs the forward pass of the DecodeBboxes layer on the NVIDIA GPU.
     * 
     * This function implements the forward pass for the DecodeBboxes layer using NVIDIA GPU
     * operations.
     */
    void Forward() override;
};

} // namespace oris_ai
