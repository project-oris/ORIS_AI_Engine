/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/convolution.h"

namespace oris_ai {

/**
 * @class DetectFeatureMap
 * @brief Represents a custom DetectFeatureMap layer in the YOLO model.
 * 
 * This class wraps a Convolution layer and manages initialization and forward pass operations
 * for the custom Detect layer in YOLO models.
 */
template <typename T>
class DetectFeatureMap : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a DetectFeatureMap layer.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     */
    DetectFeatureMap(const std::string& layer_name, Device target_device);

    /**
     * @brief Destructor for the DetectFeatureMap class.
     */
    ~DetectFeatureMap() = default;

    /**
     * @brief Initializes the DetectFeatureMap layer with the given TorchLayer data.
     * @param detect_layers Vector of TorchLayer objects for the three convolutions.
     */
    void InitDetectFeatureMap(const std::vector<TorchLayer>& detect_layers);

    /**
     * @brief Sets the input tensor for the DetectFeatureMap layer.
     * @param input_tensor The input tensor.
     */
    void SetInputTensor(Tensor<T>* input_tensor) override;

    /**
     * @brief Gets the number of input tensors.
     * 
     * @return The number of input tensors.
     */
    inline size_t GetInputSize() override { return cv2_0_->GetInputSize(); }

    /**
     * @brief Retrieves the output tensor for the DetectFeatureMap layer.
     * @return The output tensor.
     */
    Tensor<T>* GetOutputTensor() override;

    /**
     * @brief Perform the forward pass for the Conv layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { cv2_cv3_concat_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> cv2_0_;  // 1st convolution layer
    std::unique_ptr<Convolution<T>> cv2_1_;  // 2nd convolution layer
    std::unique_ptr<Convolution<T>> cv2_2_;  // 3rd convolution layer

    std::unique_ptr<Convolution<T>> cv3_0_;  // 4th convolution layer
    std::unique_ptr<Convolution<T>> cv3_1_;  // 5th convolution layer
    std::unique_ptr<Convolution<T>> cv3_2_;  // 6th convolution layer

    std::unique_ptr<Concat<T>> cv2_cv3_concat_;    // Layer to concatenate
};

} // namespace oris_ai
