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

#include "oris_ai/layer/layer_factory.h"
#include "oris_ai/model/int8/model_int8.h"
#include "oris_ai/model/yolov8/yolov8.h"

namespace oris_ai {

template <>
class Yolov8<int8_t> : public Model<int8_t> {
  public:
    /**
     * @brief Constructor to initialize Yolov8n model.
     */
    Yolov8(Device target_device, TaskType task_type);

    /**
     * @brief Destructor for Yolov8n model.
     */
    ~Yolov8() = default;

    /**
     * @brief Executes a forward pass through the Yolov8 model.
     *
     * This function processes the input tensors by passing them sequentially 
     * through each layer in the Yolov8 model.
     */
    void Forward() override;

    /**
     * @brief Performs post-processing on YOLOv8 model outputs.
     *
     * This function applies task-specific post-processing operations based on the model type:
     * - For Detection: Applies Non-Maximum Suppression to filter overlapping boxes
     * - For Segmentation: Performs both NMS and mask coefficient filtering
     *
     * @param score_threshold Minimum confidence score for a detection to be considered valid.
     * Default is 0.25.
     * @param iou_threshold Intersection over Union (IoU) threshold for suppression.
     * Default is 0.45.
     * @param max_det Maximum number of detections to keep after post-processing.
     * Default is 300.
     */
    void PostProcess(float score_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    int max_det = 300) override;

    /**
     * @brief Retrieves the detection results from the Yolov8 model.
     *
     * This function returns a vector of Detection objects containing the 
     * results from the detection layer of the model.
     *
     * @return A const reference to a vector of Detection objects representing 
     *         the results of the detection process.
     */
    inline const std::vector<Detection>& GetDetectionResults() const override {
      return model_detect_->GetDetection();
    }


  private:
    TaskType task_type_;
    std::unique_ptr<Dequant> neck_dequant_;
    std::unique_ptr<YoloDetect<float>> model_detect_;

    /**
     * @brief Parses the YOLOv8 model from a TorchModel object.
     * 
     * This function processes the parsed TorchModel object and initializes
     * the YOLOv8 model's internal structures and parameters. It sets up the
     * backbone and head layers of the YOLOv8 model.
     * 
     * @param model The TorchModel object containing the YOLOv8 model data.
     * @return true if parsing is successful, false otherwise.
     */
    bool ParsingModel(oris_ai::TorchModel& model) override;
    
    /**
     * @brief Creates and initializes a Convolution layer.
     * 
     * This function creates a Convolution layer with the specified parameters
     * and initializes it with the given input tensor and device.
     * 
     * @param index Reference to the current layer index in the model.
     * @param model The TorchModel object containing the layer parameters.
     * @param input_tensor Pointer to the input tensor for the convolution layer.
     * @param device The target device (CPU or GPU) for the layer.
     */
    void MakeConv(int& index, const TorchModel& model, Tensor<int8_t>* input_tensor, const Device& device);
};

}  // namespace oris_ai
