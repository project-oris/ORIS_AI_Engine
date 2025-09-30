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

// Basic Layers
#include "oris_ai/layer/concat.h"
#include "oris_ai/layer/convolution.h"
#include "oris_ai/layer/convolution_transpose.h"
#include "oris_ai/layer/depthwise_convolution.h"
#include "oris_ai/layer/elementwise.h"
#include "oris_ai/layer/maxpooling.h"
#include "oris_ai/layer/matmul.h"
#include "oris_ai/layer/softmax.h"
#include "oris_ai/layer/split.h"
#include "oris_ai/layer/transpose.h"
#include "oris_ai/layer/upsample.h"

// Custom Layers
#include "oris_ai/layer/custom/bottleneck.h"
#include "oris_ai/layer/custom/c2f.h"
#include "oris_ai/layer/custom/c3k.h"
#include "oris_ai/layer/custom/c3k2.h"
#include "oris_ai/layer/custom/c2psa.h"
#include "oris_ai/layer/custom/psablock.h"
#include "oris_ai/layer/custom/attention.h"
#include "oris_ai/layer/custom/decode_bboxes.h"
#include "oris_ai/layer/custom/detectfeaturemap.h"
#include "oris_ai/layer/custom/dfl.h"
#include "oris_ai/layer/custom/inverted_residual.h"
#include "oris_ai/layer/custom/mobilenet_block.h"
#include "oris_ai/layer/custom/proto.h"
#include "oris_ai/layer/custom/sppf.h"
#include "oris_ai/layer/custom/yolo_detect.h"
#include "oris_ai/layer/custom/yolo_segment.h"

// Int8 Layers
#include "oris_ai/layer/int8/concat_int8.h"
#include "oris_ai/layer/int8/convolution_int8.h"
#include "oris_ai/layer/int8/padding_int8.h"
#include "oris_ai/layer/int8/split_int8.h"
#include "oris_ai/layer/int8/dequant_int8.h"
#include "oris_ai/layer/int8/custom/bottleneck_int8.h"
#include "oris_ai/layer/int8/custom/sppf_int8.h"

namespace oris_ai {

// Basic Layer Factory Functions
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
 * @brief Factory function to create a ConvolutionTranspose layer.
 * @tparam T The data type used by the ConvolutionTranspose layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created ConvolutionTranspose layer.
 */
template <typename T>
std::unique_ptr<ConvolutionTranspose<T>> CreateConvolutionTranspose(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a DepthwiseConvolution layer.
 * @tparam T The data type used by the DepthwiseConvolution layer (e.g., float).
 * @param layer_name The name of the DepthwiseConvolution layer to be created.
 * @param target_device The device (CPU or GPU) on which the DepthwiseConvolution layer will operate.
 * @return A unique pointer to the newly created DepthwiseConvolution layer.
 */
template <typename T>
std::unique_ptr<DepthwiseConvolution<T>> CreateDepthwiseConvolution(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create an ElementWise layer.
 * @tparam T The data type used by the ElementWise layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created ElementWise layer.
 */
template <typename T>
std::unique_ptr<ElementWise<T>> CreateElementWise(const std::string& layer_name, Device target_device);

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
 * @brief Factory function to create a MatMul layer.
 * @tparam T The data type used by the MatMul layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created MatMul layer.
 */
template <typename T>
std::unique_ptr<MatMul<T>> CreateMatMul(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a Softmax layer.
 * @tparam T The data type used by the Softmax layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created Softmax layer.
 */
template <typename T>
std::unique_ptr<Softmax<T>> CreateSoftmax(const std::string& layer_name, Device target_device);

// Basic Layers without layer_name
/**
 * @brief Factory function to create a Split layer.
 * @tparam T The data type used by the Split layer (e.g., float).
 * @param layer_name The name of the Split layer to be created.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created Split layer.
 */
template <typename T>
std::unique_ptr<Split<T>> CreateSplit(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a Transpose layer.
 * @tparam T The data type used by the Transpose layer (e.g., float).
 * @param layer_name The name of the Transpose layer to be created.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created Transpose layer.
 */
template <typename T>
std::unique_ptr<Transpose<T>> CreateTranspose(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a Upsample layer.
 * @tparam T The data type used by the MaxPooling layer (e.g., float).
 * @param layer_name The name of the Upsample layer to be created.
 * @param target_device The device (CPU or GPU) on which the Upsample layer will operate.
 * @return A unique pointer to the newly created Upsample layer.
 */
template <typename T>
std::unique_ptr<Upsample<T>> CreateUpsample(const std::string& layer_name, Device target_device);

// Custom Layer Factory Functions
/**
 * @brief Factory function to create a Bottleneck layer.
 * @tparam T The data type used by the Bottleneck layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param add Flag to determine if element-wise addition is performed.
 * @return A unique pointer to the newly created Bottleneck layer.
 */
template <typename T>
std::unique_ptr<Bottleneck<T>> CreateBottleneck(const std::string& layer_name, Device target_device, bool add);

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
 * @brief Factory function to create a C3k layer.
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param shortcut Indicates whether to use a shortcut (add) operation in internal bottlenecks.
 * @return A unique pointer to the newly created C3k layer.
 */
template <typename T>
std::unique_ptr<C3k<T>> CreateC3k(const std::string& layer_name, Device target_device, bool shortcut);

/**
 * @brief Factory function to create a C3k2 layer.
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param shortcut Indicates whether to use a shortcut (add) operation in internal blocks.
 * @param c3k_count Number of C3k blocks repeated in C3k2.
 * @return A unique pointer to the newly created C3k2 layer.
 */
template <typename T>
std::unique_ptr<C3k2<T>> CreateC3k2(const std::string& layer_name, Device target_device, bool shortcut, int c3k_count);

/**
 * @brief Factory function to create a C2PSA layer.
 * @tparam T The data type used by the C2PSA layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created C2PSA layer.
 */
template <typename T>
std::unique_ptr<C2PSA<T>> CreateC2PSA(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a PSABlock layer.
 * @tparam T The data type used by the PSABlock layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created PSABlock layer.
 */
template <typename T>
std::unique_ptr<PSABlock<T>> CreatePSABlock(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create an Attention layer.
 * @tparam T The data type used by the Attention layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created Attention layer.
 */
template <typename T>
std::unique_ptr<Attention<T>> CreateAttention(const std::string& layer_name, Device target_device);
  
/**
 * @brief Factory function to create a DecodeBboxes layer.
 * @tparam T The data type used by the DecodeBboxes layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created DecodeBboxes layer.
 */
template <typename T>
std::unique_ptr<DecodeBboxes<T>> CreateDecodeBboxes(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a DetectFeatureMap layer.
 * @tparam T The data type used by the DetectFeatureMap layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param use_legacy Build a YOLOv8-style head (Conv-Conv-Conv) when true,
 *        otherwise use a YOLOv11-style head with depthwise convolutions.
 *        Defaults to false.
 * @param use_only_box_regressor If true, create only the box regressor and
 *        omit the class predictor (segmentation mode). Defaults to false.
 * @return A unique pointer to the newly created DetectFeatureMap layer.
 */
template <typename T>
std::unique_ptr<DetectFeatureMap<T>> CreateDetectFeatureMap(
  const std::string& layer_name, Device target_device,
  bool use_legacy = false, bool use_only_box_regressor = false);

/**
 * @brief Factory function to create a DFL layer.
 * @tparam T The data type used by the DFL layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created DFL layer.
 */
template <typename T>
std::unique_ptr<DFL<T>> CreateDFL(const std::string& layer_name,  Device target_device);

/**
 * @brief Factory function to create an InvertedResidual layer (MobileNet v2).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param use_expand Whether to use expand convolution.
 * @param use_residual Whether to use residual connection.
 * @return A unique pointer to the newly created InvertedResidual layer.
 */
template <typename T>
std::unique_ptr<InvertedResidual<T>> CreateInvertedResidual(
  const std::string& layer_name, Device target_device, bool use_expand, bool use_residual);

/**
 * @brief Factory function to create a MobileNetBlock layer.
 * @tparam T The data type used by the MobileNetBlock layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param out_channels Number of output channels.
 * @param expansion_factor Expansion factor for the expand convolution.
 * @param repeat_count Number of times to repeat the InvertedResidual layer.
 * @param stride Stride value for the first InvertedResidual layer.
 * @return A unique pointer to the newly created MobileNetBlock layer.
 */
template <typename T>
std::unique_ptr<MobileNetBlock<T>> CreateMobileNetBlock(
  const std::string& layer_name, Device target_device,
  size_t out_channels, size_t expansion_factor,
  size_t repeat_count, size_t stride);

/**
 * @brief Factory function to create a Proto layer.
 * @tparam T The data type used by the Proto layer (e.g., float).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created Proto layer.
 */
template <typename T>
std::unique_ptr<Proto<T>> CreateProto(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a SPPF layer (Yolo v8).
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created SPPF layer.
 */
template <typename T>
std::unique_ptr<SPPF<T>> CreateSPPF(const std::string& layer_name, Device target_device);

/**
 * @brief Factory function to create a YoloDetect layer.
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param use_legacy Use YOLOv8-style head when true, YOLOv11-style when false.
 * @return A unique pointer to the newly created YoloDetect layer.
 */
template <typename T>
std::unique_ptr<YoloDetect<T>> CreateYoloDetect(
  const std::string& layer_name, Device target_device, bool use_legacy);

/**
 * @brief Factory function to create a YoloSegment layer.
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @param use_legacy Use YOLOv8-style head when true, YOLOv11-style when false.
 * @return A unique pointer to the newly created YoloSegment layer.
 */
template <typename T>
std::unique_ptr<YoloSegment<T>> CreateYoloSegment(
  const std::string& layer_name, Device target_device, bool use_legacy);

/**
 * @brief Factory function to create a PaddingINT8 layer.
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created PaddingINT8 layer.
 */
std::unique_ptr<PaddingINT8> CreatePaddingINT8(const std::string& layer_name, Device target_device = Device::GPU);

/** 
 * @brief Factory function to create a Dequant layer.
 * @param layer_name The name of the layer.
 * @param target_device The device (CPU or GPU) on which the layer will operate.
 * @return A unique pointer to the newly created Dequant layer.
 */
std::unique_ptr<Dequant> CreateDequant(const std::string& layer_name, Device target_device = Device::GPU);

} // namespace oris_ai