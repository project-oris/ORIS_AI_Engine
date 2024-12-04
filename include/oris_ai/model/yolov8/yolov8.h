/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer_factory.h"
#include "oris_ai/model/model.h"

namespace oris_ai {

class Yolov8 : public Model {
  public:
    /**
     * @brief Constructor to initialize Yolov8n model.
     */
    Yolov8(Device target_device) : Model("yolov8n", target_device) {}

    /**
     * @brief Destructor for Yolov8n model.
     */
    ~Yolov8() = default;

    /**
     * @brief Opens the YOLOv8n model
     * 
     * @param model_path The path to the model file.
     */
    void Open(const std::string& model_path) override;

    /**
     * @brief Opens the model from an array
     * 
     * @param model_data Pointer to the binary model data.
     * @param model_size Size of the binary model data in bytes.
     */
    void OpenFromArray(const unsigned char* model_data, size_t model_size) override;

    /**
     * @brief Executes a forward pass through the Yolov8 model.
     *
     * This function processes the input tensors by passing them sequentially 
     * through each layer in the Yolov8 model.
     */
    void Forward();

    /**
     * @brief Applies Non-Maximum Suppression (NMS) to filter detection results.
     *
     * This function performs Non-Maximum Suppression to eliminate overlapping 
     * bounding boxes based on their confidence scores and IoU thresholds.
     *
     * @param score_threshold Minimum confidence score for a detection to be considered valid. Default is 0.25.
     * @param iou_threshold Intersection over Union (IoU) threshold for suppression. Default is 0.45.
     * @param max_det Maximum number of detections to keep after suppression. Default is 300.
     */
    void NonMaxSuppression(float score_threshold = 0.25f, float iou_threshold = 0.45f, int max_det = 300);

    /**
     * @brief Retrieves the detection results from the Yolov8 model.
     *
     * This function returns a vector of Detection objects containing the 
     * results from the detection layer of the model.
     *
     * @return A const reference to a vector of Detection objects representing 
     *         the results of the detection process.
     */
    inline const std::vector<Detection>& GetResult() const override {
      return model_22_detect_->GetDetectionResult();
    }

  private:
    void ParsingYolov8(oris_ai::TorchModel& model);
    bool ParsingModel(std::fstream& input);
    bool ParsingModel(std::istream& input);
    void MakeConv(int& index, const TorchModel& model, Tensor<float>* input_tensor, const Device& device);
    void SetC2fLayers(const oris_ai::TorchModel& model, std::vector<TorchLayer>& torch_layers, int& index, int bottleneck_count);
    void SetSPPFLayers(const oris_ai::TorchModel& model, std::vector<TorchLayer>& torch_layers, int& index);
    void SetDetectLayers(const oris_ai::TorchModel& model, std::vector<TorchLayer>& torch_layers, int& index);

    std::unique_ptr<Convolution<float>> model_0_conv_;
    std::unique_ptr<Convolution<float>> model_1_conv_;
    std::unique_ptr<C2f<float>> model_2_c2f_;
    std::unique_ptr<Convolution<float>> model_3_conv_;
    std::unique_ptr<C2f<float>> model_4_c2f_;
    std::unique_ptr<Convolution<float>> model_5_conv_;
    std::unique_ptr<C2f<float>> model_6_c2f_;
    std::unique_ptr<Convolution<float>> model_7_conv_;
    std::unique_ptr<C2f<float>> model_8_c2f_;
    std::unique_ptr<SPPF<float>> model_9_sppf_;
    std::unique_ptr<Upsample<float>> model_10_upsample_;
    std::unique_ptr<Concat<float>> model_11_concat_;
    std::unique_ptr<C2f<float>> model_12_c2f_;
    std::unique_ptr<Upsample<float>> model_13_upsample_;
    std::unique_ptr<Concat<float>> model_14_concat_;
    std::unique_ptr<C2f<float>> model_15_c2f_;
    std::unique_ptr<Convolution<float>> model_16_conv_;
    std::unique_ptr<Concat<float>> model_17_concat_;
    std::unique_ptr<C2f<float>> model_18_c2f_;
    std::unique_ptr<Convolution<float>> model_19_conv_;
    std::unique_ptr<Concat<float>> model_20_concat_;
    std::unique_ptr<C2f<float>> model_21_c2f_;

    std::unique_ptr<Yolov8Detect<float>> model_22_detect_;
};

}  // namespace oris_ai
