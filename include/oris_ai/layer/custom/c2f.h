/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/convolution.h"
#include "oris_ai/layer/split.h"
#include "oris_ai/layer/custom/bottleneck.h"

namespace oris_ai {

/**
 * @class C2f
 * @brief Custom C2f layer for YOLOv8n, consisting of two convolution layers and multiple bottlenecks.
 */
template <typename T>
class C2f : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize C2f layer with the given name, target device, and bottleneck count.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     * @param shortcut Indicates whether to use a shortcut (add) operation.
     * @param bottleneck_count The number of bottleneck layers to create.
     */
    C2f(const std::string& layer_name, Device target_device, bool shortcut, int bottleneck_count);

    /**
     * @brief Default destructor for the C2f class.
     */
    ~C2f() = default;

    /**
     * @brief Initializes the C2f layer with the given TorchLayer data.
     * @param c2f_layers Vector of TorchLayer objects for the two convolutions and bottlenecks.
     */
    void InitC2f(const std::vector<TorchLayer>& c2f_layers);

    /**
     * @brief Retrieves the output tensor from the last convolutional layer in the C2f layer.
     * @return The output tensor from the last convolutional layer.
     */
    inline Tensor<T>* GetOutputTensor() override { return c2f_cv2_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass for the C2f layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { c2f_cv2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> c2f_cv1_;  // First convolution layer
    std::unique_ptr<Split<T>> c2f_cv1_split_;    // Layer to concatenate before cv2
    std::vector<std::unique_ptr<Bottleneck<T>>> c2f_bottlenecks_;  // Vector of bottleneck layers
    std::unique_ptr<Convolution<T>> c2f_cv2_;  // Second convolution layer
    std::unique_ptr<Concat<T>> c2f_concat_;    // Layer to concatenate before cv2    

    bool shortcut_;  // Indicates if shortcut (add) operation is enabled
    int bottleneck_count_;  // Number of bottleneck layers

    Tensor<T>* c2f_cv1_split_0_;
    Tensor<T>* c2f_cv1_split_1_;
};

} // namespace oris_ai

