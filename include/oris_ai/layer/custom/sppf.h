/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/convolution.h"
#include "oris_ai/layer/maxpooling.h"

namespace oris_ai {

/**
 * @class SPPF
 * @brief Custom SPPF layer for YOLOv8n, consisting of convolution and max-pooling layers.
 */
template <typename T>
class SPPF : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize SPPF layer with the given name and device.
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     */
    SPPF(const std::string& layer_name, Device target_device);

    /**
     * @brief Destructor for the SPPF class.
     */
    ~SPPF() = default;

    /**
     * @brief Initializes the SPPF layer with the given TorchLayer data.
     * @param sppf_layers Vector of TorchLayer objects for the convolution and max-pooling layers.
     */
    void InitSPPF(const std::vector<TorchLayer>& sppf_layers);

    /**
     * @brief Retrieves the output tensor from the last convolutional layer in the SPPF layer.
     * @return The output tensor.
     */
    inline Tensor<T>* GetOutputTensor() override { return sppf_cv2_->GetOutputTensor(); }

    /**
     * @brief Perform the forward pass for the SPPF layer.
     */
    void Forward() override;

#ifdef USE_DEBUG_MODE
    inline void PrintOutput() override { sppf_cv2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> sppf_cv1_;  // First convolution layer (cv1)
    std::unique_ptr<MaxPooling<T>> sppf_maxpool1_;  // First max-pooling layer
    std::unique_ptr<MaxPooling<T>> sppf_maxpool2_;  // Second max-pooling layer
    std::unique_ptr<MaxPooling<T>> sppf_maxpool3_;  // Third max-pooling layer
    std::unique_ptr<Concat<T>> sppf_concat_;    // Layer to concatenate before cv2
    std::unique_ptr<Convolution<T>> sppf_cv2_;  // Second convolution layer (cv2)
};

}  // namespace oris_ai
