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
 * @brief Sets up the layers required for a C2f block.
 * 
 * This function prepares the TorchLayer objects needed for initializing
 * a C2f block, including the input convolution, bottleneck layers, and
 * output convolution.
 * 
 * @param model The TorchModel object containing the layer parameters.
 * @param torch_layers Vector to store the prepared TorchLayer objects.
 * @param index Reference to the current layer index in the model.
 * @param bottleneck_count Number of bottleneck layers in the C2f block.
 */
void SetC2fLayers(const oris_ai::TorchModel& model, std::vector<TorchLayer>& torch_layers,
                  int& index, int bottleneck_count);

/**
 * @brief Sets up the layers required for a SPPF block.
 * 
 * This function prepares the TorchLayer objects needed for initializing
 * a SPPF (Spatial Pyramid Pooling - Fast) block, including the input
 * convolution, max pooling layers, and output convolution.
 * 
 * @param model The TorchModel object containing the layer parameters.
 * @param torch_layers Vector to store the prepared TorchLayer objects.
 * @param index Reference to the current layer index in the model.
 */
void SetSPPFLayers(const oris_ai::TorchModel& model, std::vector<TorchLayer>& torch_layers, int& index);

/**
 * @brief Prepares TorchLayer objects for the detection head.
 *
 * This utility gathers the DetectFeatureMap and DFL layers required by both
 * YOLOv8 and YOLOv11 detection heads.
 * @param model The TorchModel object containing the layer parameters.
 * @param torch_layers Vector to store the prepared TorchLayer objects.
 * @param index Reference to the current layer index in the model.
 * @param model_type Model version to determine layer structure.
 */
void SetDetectLayers(const oris_ai::TorchModel& model,
                    std::vector<TorchLayer>& torch_layers,
                    int& index, ModelType model_type = ModelType::YOLOv8n);

/**
 * @brief Prepares TorchLayer objects for the segmentation head.
 *
 * This utility collects the proto layers and detection layers shared between
 * YOLOv8 and YOLOv11 segmentation heads.
 * @param model The TorchModel object containing the layer parameters.
 * @param torch_layers Vector to store the prepared TorchLayer objects.
 * @param index Reference to the current layer index in the model.
 * @param model_type Model version to determine detection head structure.
 */
void SetSegmentLayers(const oris_ai::TorchModel& model,
                      std::vector<TorchLayer>& torch_layers,
                      int& index, ModelType model_type = ModelType::YOLOv8n);

}  // namespace oris_ai
