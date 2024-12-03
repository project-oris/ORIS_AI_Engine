/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/split.h"
#include "oris_ai/layer/custom/detectfeaturemap.h"
#include "oris_ai/layer/custom/dfl.h"
#include "oris_ai/layer/custom/decode_bboxes.h"
#include "oris_ai/model/yolov8/yolov8_type.h"

namespace oris_ai {

/**
 * @class Yolov8Detect
 * @brief Represents the YOLOv8 detection layer, which combines multiple DetectFeatureMap layers.
 * 
 * This class handles the final feature maps for YOLOv8 detection by combining three DetectFeatureMap layers.
 */
template <typename T>
class Yolov8Detect : public LayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize the Yolov8Detect layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     */
    Yolov8Detect(const std::string& layer_name, Device target_device);

    /**
     * @brief Default destructor for the Yolov8Detect class.
     */
    ~Yolov8Detect() = default;

    /**
     * @brief Initializes the Yolov8Detect layer.
     * @note The actual initialization logic will be implemented later.
     */
    void InitYoloV8Detect(const std::vector<TorchLayer>& detect_layers, size_t num_classes = 80);

    /**
     * @brief Perform the forward pass for the Yolov8Detect layer.
     */
    void Forward() override;

    inline const std::vector<Detection>& GetDetectionResult() const {
      return detection_result_;
    }

    // score_threshold = conf_thres in PyTorch Yolo v8
    void ApplyNMS(float score_threshold, float iou_threshold, int max_det);

  private:
    void MakeAnchors(const std::vector<Tensor<T>*>& feature_maps, const std::array<float, 3> stride = {8.0f, 16.0f, 32.0f}, float grid_cell_offset = 0.5f);

    void Sigmoid();

    // Three DetectFeatureMap objects
    std::unique_ptr<DetectFeatureMap<T>> detect_feature_map_0_;
    std::unique_ptr<DetectFeatureMap<T>> detect_feature_map_1_;
    std::unique_ptr<DetectFeatureMap<T>> detect_feature_map_2_;
    std::unique_ptr<Concat<T>> feature_map_concat_;
    std::unique_ptr<Split<T>> feature_map_concat_to_box_cls_;
    std::unique_ptr<DFL<T>> dfl_;
    std::unique_ptr<DecodeBboxes<T>> decode_bboxes_;  // get dbox

    // Anchor points and stride tensors as 2D Tensors
    std::unique_ptr<Tensor<T>> anchor_points_;  // Tensor for anchor points
    std::unique_ptr<Tensor<T>> strides_;  // Tensor for stride values
    
    // std::vector<size_t> box_cls_split_sizes_; // Size to split box and cls tensors
    Tensor<T>* box_;
    Tensor<T>* cls_;

    std::vector<Detection> detection_result_;
};

}  // namespace oris_ai
