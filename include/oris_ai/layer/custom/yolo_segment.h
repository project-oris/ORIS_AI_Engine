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
#include "oris_ai/layer/custom/proto.h"
#include "oris_ai/layer/custom/detectfeaturemap.h"
#include "oris_ai/layer/custom/yolo_detect.h"

namespace oris_ai {

/**
 * @class YoloSegment
 * @brief Represents the YOLO segmentation layer, which combines segmentation capabilities (proto. cv4) with the YOLO detection layer.
 *
 * This class handles the final feature maps for YOLO segmentation.
 */
template <typename T>
class YoloSegment : public YoloDetect<T> {
  public:
    /**
     * @brief Constructor to initialize the YoloSegment layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param use_legacy Use YOLOv8-style head when true, YOLOv11-style when false.
     */
    YoloSegment(const std::string& layer_name, Device target_device, bool use_legacy);

    /**
     * @brief Default destructor for the YoloSegment class.
     */
    ~YoloSegment() = default;

    /**
     * @brief Initializes the YoloSegment layer.
     * @param segment_layers Vector of TorchLayer objects containing the layer parameters.
     */
    void InitYoloSegment(const std::vector<TorchLayer>& segment_layers);

    /**
     * @brief Perform the forward pass for the YoloSegment layer.
     * 
     * This function extends the detection forward pass with segmentation processing.
     */
    void Forward() override;

    /**
     * @brief Apply Non-Maximum Suppression (NMS) to the detection and segmentation results.
     * @param score_threshold The score threshold for the detection.
     * @param iou_threshold The IoU threshold for the detection.
     * @param max_det The maximum number of detections.
     */
    void ApplyNMSCPU(float score_threshold, float iou_threshold, int max_det) override;

    /**
     * @brief Process masks using proto features and mask coefficients
     */
    void ProcessMasks();

    inline const std::vector<float>& GetMask() const {
      return final_masks_;
    }

  protected:
    /**
     * @brief Store detection and segmentation results after NMS
     * @param filtered_results Source vector containing detection and mask coefficient data
     * @param indices NMS result indices
     */
    void StoreNMSResults(
      const std::vector<std::vector<T>>& filtered_results,
      const std::vector<int>& indices) override;

  private:
    std::unique_ptr<Proto<T>> proto_;
    std::unique_ptr<DetectFeatureMap<T>> cv4_0_;
    std::unique_ptr<DetectFeatureMap<T>> cv4_1_;
    std::unique_ptr<DetectFeatureMap<T>> cv4_2_;
    std::unique_ptr<Concat<T>> mc_;  // Concatenated mask coefficients from cv4 feature maps
    std::vector<MaskCoefficients> mask_coefficients_;  // Stores mask coefficients for each detected object after NMS

    size_t input_height_;  // Original input image height
    size_t input_width_;   // Original input image width
    std::vector<float> final_masks_;  // Final upsampled and normalized masks

    /**
     * @brief Compute mask features by matrix multiplication of mask coefficients and proto features
     * @return Vector containing the computed mask features
     */
    std::vector<float> ComputeMaskFeatures() const;

    /**
     * @brief Crop masks using bounding boxes
     * @param masks Input/output vector containing mask data to crop
     * @param downsampled_boxes Scaled bounding boxes for cropping
     * @param height Height of each mask
     * @param width Width of each mask
     */
    void CropMasks(std::vector<float>& masks, const std::vector<std::vector<T>>& downsampled_boxes,
      size_t height, size_t width) const;

    /**
     * @brief Upsample and normalize segmentation masks
     * @param input_masks Input vector containing mask data to process
     * @param in_h Height of input masks
     * @param in_w Width of input masks
     * @param channels Number of masks to process
     */
    void UpsampleAndNormalizeMasks(const std::vector<float>& input_masks, int in_h, int in_w, int channels);
};

}  // namespace oris_ai 