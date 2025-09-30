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

#include "oris_ai/model/yolov8/yolov8.h"

namespace oris_ai {

template<typename T>
class MobileNetYolov8 : public Yolov8<T> {
  public:
    /**
     * @brief Constructor to initialize MobileNetYolov8 model.
     * @param target_device The target device (CPU or GPU) on which the model will run.
     * @param task_type The type of task the model will perform.
     */
    MobileNetYolov8(Device target_device, TaskType task_type) : Yolov8<T>(target_device, task_type) {}

    /**
     * @brief Destructor for MobileNetYolov8 model.
     */
    ~MobileNetYolov8() = default;

    /**
     * @brief Executes a forward pass through the MobileNetYolov8 model.
     *
     * This function processes the input tensors by passing them sequentially 
     * through each layer in the MobileNetYolov8 model.
     */
    void Forward() override;

  private:
    /**
     * @brief Sets up MobileNet block layers with specified parameters.
     * 
     * @param model TorchModel object containing the model data
     * @param torch_layers Vector to store the TorchLayer objects
     * @param index Current index in the model layers
     * @param out_channels Output channels for the block
     * @param expansion Expansion factor for the block
     * @param repeat Number of times to repeat the block
     * @param stride Stride value for the block
     */
    void SetMobilenetBlockLayers(oris_ai::TorchModel& model, std::vector<TorchLayer>& torch_layers,
                                int& index, int out_channels, int expansion, int repeat, int stride);

    /**
     * @brief Parses the MobileNetYolov8 model from a TorchModel object.
     * 
     * This function processes the parsed TorchModel object and initializes
     * the MobileNetYolov8 model's internal structures and parameters. It sets up the
     * backbone and head layers of the MobileNetYolov8 model.
     * 
     * @param model The TorchModel object containing the MobileNetYolov8 model data.
     * @return true if parsing is successful, false otherwise.
     */
    bool ParsingModel(oris_ai::TorchModel& model) override;
};

}  // namespace oris_ai 