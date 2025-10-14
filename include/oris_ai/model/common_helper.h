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
#include "oris_ai_model.pb.h"

#include <memory>
#include <vector>

namespace oris_ai {

/**
 * @brief Creates and initializes a Convolution layer (Conv2d + Activation).
 *
 * This is a common pattern used across various neural network architectures
 * (YOLO, ResNet, EfficientNet, etc.). It processes two consecutive layers
 * from the protobuf model: TorchConv2d and TorchActivation.
 *
 * @param model The TorchModel object containing the layer parameters.
 * @param layers Vector to which the created layer will be added.
 * @param index Reference to the current layer index in the model (incremented by 2).
 * @param input_tensor Pointer to the input tensor for the convolution layer.
 * @param device The target device (CPU or GPU) for the layer.
 */
template<typename T>
void MakeConv(const TorchModel& model,
  std::vector<std::unique_ptr<HiddenLayerAbstract<T>>>& layers,
  int& index, Tensor<T>* input_tensor,const Device& device);

}  // namespace oris_ai
