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

#include "oris_ai/model/model.h"

namespace oris_ai {

/**
 * @brief Base class for YOLO family models providing common preprocessing functionality.
 *
 * This class implements YOLO-style preprocessing (normalization by 255) and serves as
 * a base for all YOLO variants (YOLOv8, YOLOv11, MobileNet-YOLO, etc.).
 */
template<typename T>
class YoloBase : public Model<T> {
  public:
    /**
     * @brief Constructor to initialize YoloBase with target device.
     */
    YoloBase(Device target_device) : Model<T>(target_device) {}

    /**
     * @brief Virtual destructor for YoloBase class.
     */
    virtual ~YoloBase() = default;

    /**
     * @brief Converts and sets an input image as a tensor in NCHW format with YOLO-style normalization.
     * This function automatically selects CPU or GPU processing based on the target device.
     *
     * YOLO preprocessing: Normalize by dividing by 255.0
     *
     * @param input_image The input image as a `cv::Mat` object. It must be a valid 3-channel
     * image matrix in BGR format by default.
     */
    void SetInputImageTensor(const cv::Mat& input_image) override;

  protected:
    /**
     * @brief Converts and sets an input image as a tensor in NCHW format with normalization.
     * This function processes an input image in OpenCV's `cv::Mat` format and prepares it for
     * use as a model input using CPU-based processing.
     *
     * @param input_image The input image as a `cv::Mat` object. It must be a valid 3-channel
     * image matrix in BGR format by default.
     */
    void SetInputImageTensorCPU(const cv::Mat& input_image);

    /**
     * @brief Converts and sets an input image as a tensor in NCHW format with normalization.
     * This function processes an input image in OpenCV's `cv::Mat` format and prepares it for
     * use as a model input using GPU-based processing.
     *
     * @param input_image The input image as a `cv::Mat` object. It must be a valid 3-channel
     * image matrix in BGR format by default.
     */
    void SetInputImageTensorGPU(const cv::Mat& input_image);
};

}  // namespace oris_ai
