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
#include "oris_ai/model/yolo_base.h"
#include "oris_ai/model/task_type.h"

namespace oris_ai {

template<typename T>
class Yolov11 : public YoloBase<T> {
  public:
    /**
     * @brief Constructor to initialize Yolov11 model.
     */
    Yolov11(Device target_device, TaskType task_type);

    /**
     * @brief Destructor for Yolov11 model.
     */
    ~Yolov11() = default;

    /**
     * @brief Executes a forward pass through the Yolov11 model.
     *
     * This function processes the input tensors by passing them sequentially
     * through each layer in the Yolov11 model.
     */
    void Forward() override;

    /**
     * @brief Performs post-processing on YOLOv11 model outputs.
     *
     * This function applies task-specific post-processing operations based on
     * the model type:
     * - For Detection: Applies Non-Maximum Suppression to filter overlapping boxes
     * - For Segmentation: Performs both NMS and mask coefficient filtering
     *
     * @param score_threshold Minimum confidence score for a detection to be
     *                        considered valid. Default is 0.25.
     * @param iou_threshold Intersection over Union (IoU) threshold for
     *                      suppression. Default is 0.45.
     * @param max_det Maximum number of detections to keep after
     *                post-processing. Default is 300.
     */
    void PostProcess(float score_threshold = 0.25f,
                    float iou_threshold = 0.45f,
                    int max_det = 300) override;


    /**
     * @brief Retrieves the detection results from the Yolov11 model.
     *
     * @return A const reference to a vector of Detection objects representing
     *         the results of the detection process.
     */
    inline const std::vector<Detection>& GetDetectionResults() const override {
      if (task_type_ == TaskType::Detection) {
        return model_detect_->GetDetection();
      } else {
        return model_segment_->GetDetection();
      }
    }

    /**
     * @brief Retrieves the segmentation mask from the Yolov11 model.
     *
     * @return A const reference to a vector of float values representing the
     *         segmentation mask.
     */
    inline const std::vector<float>& GetSegmentationMask() const override {
      return model_segment_->GetMask();
    }

  protected:
    /**
     * @brief Parses the YOLOv11 model from a TorchModel object.
     *
     * @param model The TorchModel object containing the YOLOv11 model data.
     * @return true if parsing is successful, false otherwise.
     */
    bool ParsingModel(oris_ai::TorchModel& model) override;

  private:
    TaskType task_type_;
    std::unique_ptr<YoloDetect<T>> model_detect_;
    std::unique_ptr<YoloSegment<T>> model_segment_;
};

}  // namespace oris_ai
