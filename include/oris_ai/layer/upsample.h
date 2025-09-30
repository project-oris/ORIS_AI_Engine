/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute (ETRI)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

namespace oris_ai {

// Define the UpsampleMode enum class
enum class UpsampleMode {
  NEAREST,
  BILINEAR
};

/**
 * @class Upsample
 * @brief Represents an upsampling layer in a neural network.
 * 
 * This class defines an upsampling layer.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Upsample : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize an upsampling layer.
     * @param name The name of the layer.
     */
    Upsample(const std::string& layer_name)
      : HiddenLayerAbstract<T>(layer_name),
        scale_factor_(2.0),
        mode_(UpsampleMode::NEAREST) {}

    /**
     * @brief Destructor for the Upsample class.
     */
    ~Upsample() = default;

    /**
     * @brief Initializes the upsampling layer with parameters from a TorchUpsample object.
     * 
     * @param upsample The TorchUpsample object.
     */
    void InitUpsample(const TorchUpsample& upsample);

    /**
     * @brief Pure virtual function to perform the forward pass of the upsampling layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the upsample operation.
     */
    virtual void Forward() = 0;

  protected:
    float scale_factor_; // The scaling factor for upsampling
    // std::string mode_;   // Upsample mode (e.g., "nearest", "bilinear")
    UpsampleMode mode_;  // Upsample mode (e.g., "nearest", "bilinear")
    size_t output_height_, output_width_; // Output dimensions after the upsampling operation
};

} // namespace oris_ai
