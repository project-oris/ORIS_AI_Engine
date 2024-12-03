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
    Yolov8(Device target_device) : Model("yolov8n", target_device, 640, 640) {}

    /**
     * @brief Destructor for Yolov8n model.
     */
    ~Yolov8() = default;

    /**
     * @brief Opens the YOLOv8n model
     * 
     * @param model_path The path to the model file.
     * @return bool indicating success or failure.
     */
    bool Open(const std::string& model_path) override;

    /**
     * @brief Executes a forward pass through the Yolov8 model.
     *
     * This function processes the input tensors by passing them sequentially 
     * through each layer in the Yolov8 model.
     */
    void Forward();

    void NonMaxSuppression(float score_threshold = 0.25f, float iou_threshold = 0.45f, int max_det = 300);

    inline const std::vector<Detection>& GetResult() const override {
      return model_22_detect_->GetDetectionResult();
    }

  private:
    bool ParsingModel(std::fstream& input);
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
