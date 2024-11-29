/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/convolution.h"
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
     * @brief Destructor for the C2f class.
     */
    ~C2f();

    /**
     * @brief Initializes the C2f layer with the given TorchLayer data.
     * @param c2f_layers Vector of TorchLayer objects for the two convolutions and bottlenecks.
     */
    void InitC2f(const std::vector<TorchLayer>& c2f_layers);

    /**
     * @brief Sets the input tensor for the C2f layer.
     * @param input_tensor The input tensor.
     */
    void SetInputTensor(Tensor<T>* input_tensor) override;

    /**
     * @brief Gets the number of input tensors.
     * 
     * @return The number of input tensors.
     */
    inline size_t GetInputSize() override { return c2f_cv1_->GetInputSize(); }

    /**
     * @brief Retrieves the output tensor for the C2f layer.
     * @return The output tensor.
     */
    Tensor<T>* GetOutputTensor() override;

    /**
     * @brief Perform the forward pass for the C2f layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { c2f_cv2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> c2f_cv1_;  // First convolution layer
    std::vector<std::unique_ptr<Bottleneck<T>>> c2f_bottlenecks_;  // Vector of bottleneck layers
    std::unique_ptr<Convolution<T>> c2f_cv2_;  // Second convolution layer
    std::unique_ptr<Concat<T>> c2f_concat_;    // Layer to concatenate before cv2

    bool shortcut_;  // Indicates if shortcut (add) operation is enabled
    int bottleneck_count_;  // Number of bottleneck layers

    std::vector<Tensor<T>*> split_tensors_;  // Tensors created by splitting cv1 output
    // std::vector<const Tensor<T>*> tensors_to_concat_;  // Tensors to concatenate before cv2
    // Tensor<T>* concat_tensor_;  // Concatenated tensor after bottlenecks
};

} // namespace oris_ai

