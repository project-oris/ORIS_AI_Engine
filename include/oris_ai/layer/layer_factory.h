/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/convolution.h"
#include "oris_ai/layer/maxpooling.h"
#include "oris_ai/layer/upsample.h"

#include "oris_ai/layer/custom/c2f.h"
#include "oris_ai/layer/custom/sppf.h"
#include "oris_ai/layer/custom/yolov8_detect.h"
// #include "oris_ai/layer/custom/detectfeaturemap.h"

namespace oris_ai {

  /**
 * @brief Factory function to create a Concat layer.
 * @tparam T The data type used by the Concat layer (e.g., float).
 * @param layer_name The name of the Concat layer to be created.
 * @param target_device The device (CPU or GPU) on which the Concat layer will operate.
 * @return A unique pointer to the newly created Concat layer.
 */
template <typename T>
std::unique_ptr<Concat<T>> CreateConcat(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a Convolution layer.
 * @tparam T The data type used by the Convolution layer (e.g., float).
 * @param layer_name The name of the Convolution layer to be created.
 * @param target_device The device (CPU or GPU) on which the Convolution layer will operate.
 * @return A unique pointer to the newly created Convolution layer.
 */
template <typename T>
std::unique_ptr<Convolution<T>> CreateConvolution(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a MaxPooling layer.
 * @tparam T The data type used by the MaxPooling layer (e.g., float).
 * @param layer_name The name of the MaxPooling layer to be created.
 * @param target_device The device (CPU or GPU) on which the MaxPooling layer will operate.
 * @return A unique pointer to the newly created MaxPooling layer.
 */
template <typename T>
std::unique_ptr<MaxPooling<T>> CreateMaxPooling(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a Upsample layer.
 * @tparam T The data type used by the MaxPooling layer (e.g., float).
 * @param layer_name The name of the Upsample layer to be created.
 * @param target_device The device (CPU or GPU) on which the Upsample layer will operate.
 * @return A unique pointer to the newly created Upsample layer.
 */
template <typename T>
std::unique_ptr<Upsample<T>> CreateUpsample(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a C2f layer (Yolo v8).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param shortcut Indicates whether to use a shortcut (add) operation.
 * @param bottleneck_count The number of bottleneck layers to create.
 * @return A unique pointer to the newly created C2f layer.
 */
template <typename T>
std::unique_ptr<C2f<T>> CreateC2f(const std::string& layer_name, Device target_device, bool shortcut, int bottleneck_count);

/**
 * @brief Factory function to create a SPPF layer (Yolo v8).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created SPPF layer.
 */
template <typename T>
std::unique_ptr<SPPF<T>> CreateSPPF(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a DetectFeatureMa layer (Yolo v8).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created DetectFeatureMa layer.
 */
// template <typename T>
// std::unique_ptr<DetectFeatureMap<T>> CreateDetectFeatureMap(const std::string& layer_name, Device target_device);

template <typename T>
std::unique_ptr<Yolov8Detect<T>> CreateYolov8Detect(const std::string& layer_name, Device target_device);

} // namespace oris_ai