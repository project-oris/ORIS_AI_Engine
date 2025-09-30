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
#include "oris_ai/layer/convolution.h"
#include "oris_ai/layer/depthwise_convolution.h"

namespace oris_ai {

template <typename T>
class FeatureMapBoxRegressor;

template <typename T>
class FeatureMapClassPredictor;

/**
 * @class DetectFeatureMap
 * @brief Custom DetectFeatureMap layer for YOLO models, consisting of box regressor and class predictor.
 */
template <typename T>
class DetectFeatureMap : public HiddenLayerAbstract<T> {
  public:
  /**
   * @brief Constructor to initialize DetectFeatureMap layer.
   * @param layer_name The name of the layer.
   * @param target_device The target device (CPU or GPU) on which the layer will run.
   * @param use_legacy Use YOLOv8-style head when true, YOLOv11-style when false.
   * @param use_only_box_regressor If true, omit class predictor and concatenation (segmentation).
   */
  DetectFeatureMap(const std::string& layer_name, Device target_device,
                  bool use_legacy, bool use_only_box_regressor);

  /**
   * @brief Destructor for the DetectFeatureMap class.
   */
  ~DetectFeatureMap() = default;

  /**
   * @brief Initializes the DetectFeatureMap layer with the given TorchLayer data.
   * @param layers Vector of TorchLayer objects for the step and head components.
   */
  void InitDetectFeatureMap(const std::vector<TorchLayer>& layers);

  /**
   * @brief Retrieves the output tensor from the DetectFeatureMap layer.
   * @return The output tensor from the DetectFeatureMap layer.
   */
  inline Tensor<T>* GetOutputTensor() override {
    return use_only_box_regressor_ ?
      cv2_->GetOutputTensor()      // Segmentation
      : concat_layer_->GetOutputTensor();  // Detection
  }

  /**
   * @brief Perform the forward pass for the DetectFeatureMap layer.
   */
  void Forward() override;

#ifdef USE_DEBUG_MODE
  /**
   * @brief Prints output tensor in NCHW format.
   */
  inline void PrintOutput() override {
    if (use_only_box_regressor_) {  // Segmentation
      cv2_->PrintOutput();
    } else {  // Detection
      concat_layer_->PrintOutput();
    }
  }
#endif

  private:
    bool use_only_box_regressor_;                               // Use only box regressor (segmentation)
    std::unique_ptr<FeatureMapBoxRegressor<T>>   cv2_;          // Box regressor layer
    std::unique_ptr<FeatureMapClassPredictor<T>> cv3_;          // Class predictor layer
    std::unique_ptr<Concat<T>>                   concat_layer_; // Concatenation layer
};

/**
 * @class FeatureMapBoxRegressor
 * @brief Shared three-convolution front-end for detection and segmentation.
 */
template <typename T>
class FeatureMapBoxRegressor : public HiddenLayerAbstract<T> {
  public:
  /**
   * @brief Constructor to initialize FeatureMapBoxRegressor.
   * @param target_device The target device (CPU or GPU) on which the layer will run.
   */
  explicit FeatureMapBoxRegressor(Device target_device);

  /**
   * @brief Destructor for the FeatureMapBoxRegressor class.
   */
  ~FeatureMapBoxRegressor() = default;

  /**
   * @brief Initializes the three convolution layers with the given TorchLayer data.
   * @param layers Vector of TorchLayer objects for the convolutions.
   */
  void Init(const std::vector<TorchLayer>& layers);

  /**
   * @brief Retrieves the output tensor from the last convolution.
   * @return The output tensor from the last convolution.
   */
  inline Tensor<T>* GetOutputTensor() override { return conv_2_->GetOutputTensor(); }

  /**
   * @brief Perform the forward pass for the FeatureMapBoxRegressor layer.
   */
  void Forward() override;

#ifdef USE_DEBUG_MODE
  inline void PrintOutput() override { conv_2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> conv_0_;  // First convolution layer
    std::unique_ptr<Convolution<T>> conv_1_;  // Second convolution layer
    std::unique_ptr<Convolution<T>> conv_2_;  // Third convolution layer
};

/**
 * @class FeatureMapClassPredictor
 * @brief Abstract base class for version-specific class prediction heads.
 */
template <typename T>
class FeatureMapClassPredictor : public HiddenLayerAbstract<T> {
  public:
  /**
   * @brief Constructor to initialize FeatureMapClassPredictor.
   * @param target_device The target device (CPU or GPU) on which the layer will run.
   */
  explicit FeatureMapClassPredictor(Device target_device)
    : HiddenLayerAbstract<T>(""), target_device_(target_device) {}

  /**
   * @brief Destructor for the FeatureMapClassPredictor class.
   */
  virtual ~FeatureMapClassPredictor() = default;

  /**
   * @brief Initializes the class prediction head with the given TorchLayer data.
   * @param layers Vector of TorchLayer objects for the head.
   */
  virtual void Init(const std::vector<TorchLayer>& layers) = 0;

  /**
   * @brief Retrieves the output tensor.
   * @return The output tensor.
   */
  virtual Tensor<T>* GetOutputTensor() override = 0;

  /**
   * @brief Perform the forward pass for the class prediction head.
   */
  virtual void Forward() override = 0;

#ifdef USE_DEBUG_MODE
  virtual void PrintOutput() override = 0;
#endif

  protected:
    Device target_device_;  // Target device (CPU or GPU)
};

/**
 * @class FeatureMapClassPredictorV8
 * @brief YOLOv8 class predictor: Conv -> Conv -> Conv.
 */
template <typename T>
class FeatureMapClassPredictorV8 : public FeatureMapClassPredictor<T> {
  public:
  /**
   * @brief Constructor to initialize FeatureMapClassPredictorV8.
   * @param target_device The target device (CPU or GPU) on which the layer will run.
   */
  explicit FeatureMapClassPredictorV8(Device target_device);

  /**
   * @brief Destructor for the FeatureMapClassPredictorV8 class.
   */
  ~FeatureMapClassPredictorV8() = default;

  void Init(const std::vector<TorchLayer>& layers) override;

  inline Tensor<T>* GetOutputTensor() override { return conv_2_->GetOutputTensor(); }
  void Forward() override;

#ifdef USE_DEBUG_MODE
  inline void PrintOutput() override { conv_2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<Convolution<T>> conv_0_;  // First convolution layer
    std::unique_ptr<Convolution<T>> conv_1_;  // Second convolution layer
    std::unique_ptr<Convolution<T>> conv_2_;  // Third convolution layer
};

/**
 * @class FeatureMapClassPredictorV11
 * @brief YOLOv11 class predictor: DWConv -> Conv -> DWConv -> Conv -> Conv.
 */
template <typename T>
class FeatureMapClassPredictorV11 : public FeatureMapClassPredictor<T> {
  public:
  /**
   * @brief Constructor to initialize FeatureMapClassPredictorV11.
   * @param target_device The target device (CPU or GPU) on which the layer will run.
   */
  explicit FeatureMapClassPredictorV11(Device target_device);

  /**
   * @brief Destructor for the FeatureMapClassPredictorV11 class.
   */
  ~FeatureMapClassPredictorV11() = default;

  void Init(const std::vector<TorchLayer>& layers) override;

  inline Tensor<T>* GetOutputTensor() override { return conv_2_->GetOutputTensor(); }

  void Forward() override;

#ifdef USE_DEBUG_MODE
  inline void PrintOutput() override { conv_2_->PrintOutput(); }
#endif

  private:
    std::unique_ptr<DepthwiseConvolution<T>> dw_conv_0_;  // First depthwise convolution layer
    std::unique_ptr<Convolution<T>>          conv_0_;     // First convolution layer
    std::unique_ptr<DepthwiseConvolution<T>> dw_conv_1_;  // Second depthwise convolution layer
    std::unique_ptr<Convolution<T>>          conv_1_;     // Second convolution layer
    std::unique_ptr<Convolution<T>>          conv_2_;     // Third convolution layer
};

}  // namespace oris_ai

