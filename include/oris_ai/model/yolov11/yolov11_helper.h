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

#include "oris_ai/model/model.h"

namespace oris_ai {

/**
 * @brief Sets up the layers required for a C3k2 block.
 *
 * This function prepares the TorchLayer objects needed for initializing
 * a C3k2 block, including the input convolution, repeated C3k modules,
 * and output convolution.
 *
 * @param model The TorchModel object containing the layer parameters.
 * @param torch_layers Vector to store the prepared TorchLayer objects.
 * @param index Reference to the current layer index in the model.
 * @param c3k_count Number of C3k modules inside the C3k2 block.
 */
void SetC3k2Layers(const oris_ai::TorchModel& model,
  std::vector<TorchLayer>& torch_layers,
  int& index, int c3k_count);

/**
* @brief Sets up the layers required for a C2PSA block.
*
* This function prepares the TorchLayer objects needed for initializing
* a C2PSA block, including the first convolution layer, a single
* PSABlock, and the final convolution layer. The PSABlock consists of qkv
* convolution and activation, positional-encoding depthwise convolution and
* activation, projection convolution and activation, and two bottleneck
* convolutions with their corresponding activations.
*
* @param model The TorchModel object containing the layer parameters.
* @param torch_layers Vector to store the prepared TorchLayer objects.
* @param index Reference to the current layer index in the model.
*/
void SetC2PSALayers(const oris_ai::TorchModel& model,
  std::vector<TorchLayer>& torch_layers,
  int& index);

}  // namespace oris_ai
