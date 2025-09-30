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

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/split.h"
#include "oris_ai/layer/custom/detectfeaturemap.h"
#include "oris_ai/layer/custom/dfl.h"
#include "oris_ai/layer/custom/decode_bboxes.h"
#include "oris_ai/model/yolov8/yolov8_result.h"
#include <opencv2/opencv.hpp>

namespace oris_ai {

/**
 * @class YoloDetect
 * @brief Represents the YOLO detection layer, which combines multiple DetectFeatureMap layers.
 *
 * This class handles the final feature maps for YOLO detection by combining three DetectFeatureMap layers.
 */
template <typename T>
class YoloDetect : public LayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize the YoloDetect layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param use_legacy Use YOLOv8-style head when true, YOLOv11-style when false.
     */
    YoloDetect(const std::string& layer_name, Device target_device, bool use_legacy);

    /**
     * @brief Default destructor for the YoloDetect class.
     */
    ~YoloDetect() = default;

    /**
     * @brief Initializes the YoloDetect layer.
     * @param detect_layers Vector of TorchLayer objects containing the layer parameters.
     */
    void InitYoloDetect(const std::vector<TorchLayer>& detect_layers);

    /**
     * @brief Perform the forward pass for the YoloDetect layer.
     */
    void Forward() override;

    /**
     * @brief Get the detection result.
     * @return const std::vector<Detection>& The detection result.
     */
    inline const std::vector<Detection>& GetDetection() const {
      return detection_result_;
    }

    /**
     * @brief Apply Non-Maximum Suppression (NMS) to the detection result.
     * @param score_threshold The score threshold for the detection (conf_thres in PyTorch Yolo v8).
     * @param iou_threshold The IoU threshold for the detection.
     * @param max_det The maximum number of detections.
     */
    virtual void ApplyNMSCPU(float score_threshold, float iou_threshold, int max_det);

    // score_threshold = conf_thres in PyTorch Yolo v8
    // void ApplyNMSGPU(float score_threshold, float iou_threshold, int max_det);

    // NMS 결과의 인덱스를 반환하는 함수 추가
    // const std::vector<int>& GetNMSIndices() const { return nms_indices_; }

    /**
     * @brief Get the decoded bounding boxes output tensor
     * @return Pointer to the decoded bounding boxes tensor
     */
    // inline const Tensor<T>* GetDecodeBboxesOutput() const {
    //   return decode_bboxes_->GetOutputTensor();
    // }

    /**
     * @brief Get the classification output tensor
     * @return Pointer to the classification tensor
     */
    // inline const Tensor<T>* GetClsOutput() const {
    //   return cls_;
    // }

  protected:
    /**
     * @brief Creates anchor points and strides for the YOLOv8 detection model.
     * @param feature_maps Vector of feature map tensors to generate anchors from.
     * @param stride Array of stride values for each feature map level (default: {8.0f, 16.0f, 32.0f}).
     * @param grid_cell_offset Offset value for grid cell centers (default: 0.5f).
     */
    void MakeAnchors(const std::vector<Tensor<T>*>& feature_maps, 
                    const std::array<float, 3> stride = {8.0f, 16.0f, 32.0f}, 
                    float grid_cell_offset = 0.5f);

    /**
     * @brief Applies the sigmoid activation function to the classification tensor.
     * 
     * This function processes the classification tensor (cls_) using either CPU-based
     * Eigen operations or CUDA-based operations depending on the device type.
     */
    void Sigmoid();

    /**
     * @brief Common function to calculate and set view shapes for tensors
     * @param feature_maps Vector of feature map layers to process
     */
    void SetupFeatureMapViewShapes(const std::vector<HiddenLayerAbstract<T>*>& feature_maps);

    /**
     * @brief Find maximum confidence and class index for a detection
     * @param cls_data Pointer to classification data
     * @param num_classes Number of classes
     * @param detection_idx Index of current detection
     * @param num_detections Total number of detections
     * @return std::pair<T, int> Pair of (max_confidence, class_index)
     */
    std::pair<T, int> FindMaxConfidence(
      const T* cls_data,
      const int num_classes,
      const int detection_idx,
      const int num_detections) const;

    /**
     * @brief Process filtered results to limit maximum number and sort by confidence
     * @param filtered_results Vector of filtered detection results
     * @param max_nms Maximum number of results to keep (default: 30000)
     */
    void ProcessFilteredResults(
      std::vector<std::vector<T>>& filtered_results,
      const size_t max_nms = 30000) const;

    /**
     * @brief Prepare boxes and scores for NMS
     * @param filtered_results Vector of filtered detection results
     * @param boxes Output vector of OpenCV rectangles
     * @param scores Output vector of confidence scores
     */
    void PrepareNMSInputs(
      const std::vector<std::vector<T>>& filtered_results,
      std::vector<cv::Rect>& boxes,
      std::vector<float>& scores) const;

    /**
     * @brief Store detection results after NMS
     * @param filtered_results Source vector containing detection data
     * @param indices NMS result indices
     */
    virtual void StoreNMSResults(
      const std::vector<std::vector<T>>& filtered_results,
      const std::vector<int>& indices);

    // Protected member variables
    std::unique_ptr<DetectFeatureMap<T>> detect_feature_map_0_;
    std::unique_ptr<DetectFeatureMap<T>> detect_feature_map_1_;
    std::unique_ptr<DetectFeatureMap<T>> detect_feature_map_2_;
    std::unique_ptr<Concat<T>> feature_map_concat_;
    std::unique_ptr<Split<T>> feature_map_concat_to_box_cls_;
    std::unique_ptr<DFL<T>> dfl_;
    std::unique_ptr<DecodeBboxes<T>> decode_bboxes_;

    std::unique_ptr<Tensor<T>> anchor_points_;
    std::unique_ptr<Tensor<T>> strides_;
    
    Tensor<T>* box_;
    Tensor<T>* cls_;

    std::vector<Detection> detection_result_;
};

}  // namespace oris_ai
