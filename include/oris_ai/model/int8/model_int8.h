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

template <>
class Model<int8_t> {
  public:
    /**
     * @brief Constructor to initialize a Model with a given name.
     */
    Model(Device target_device) : target_device_(target_device), input_height_(0), input_width_(0), normalization_value_(1.0f), gpu_input_data_(nullptr) {}

    /**
     * @brief Virtual destructor for the Model class.
     * Frees GPU memory if it was allocated.
     */
    virtual ~Model();

    /**
     * @brief Open the model.
     * 
     * @param model_path The path to the model file.
     */
    void Open(const std::string& model_path);

    /**
     * @brief Opens the model from an array
     * 
     * @param model_data Pointer to the binary model data.
     * @param model_size Size of the binary model data in bytes.
     */
    // void OpenFromArray(const unsigned char* model_data, size_t model_size);

    /**
     * @brief Creates the input tensor with the specified height, width, and normalization
     * value.
     *
     * This function initializes the input tensor to the specified dimensions and prepares it
     * for subsequent use in the model. If the input tensor is not already created, it
     * allocates memory for the tensor based on the input shape (NCHW format) and fills it
     * with a default value (e.g., zeros).
     * The normalization value is stored for later use in pixel value normalization during
     * input preprocessing.
     *
     * @param input_height The height of the input tensor in pixels.
     * @param input_width The width of the input tensor in pixels.
     * @param normalization_value The value used for normalizing the pixel values. Typically
     * set to 255.0 for images with pixel values in the 0-255 range.
     */
    void CreateInputImageTensor(size_t input_height, size_t input_width, float normalization_value);

    /**
     * @brief Converts and sets an input image as a tensor in NHWC format with normalization and quantization.
     * This function processes an input image in OpenCV's `cv::Mat` format and prepares it for
     * use as a model input using GPU-based processing.
     *
     * @param input_image The input image as a `cv::Mat` object. It must be a valid 3-channel
     * image matrix in BGR format by default.
     */
    void SetInputImageTensor(const cv::Mat& input_image);

    /**
     * @brief Parses the model from a TorchModel object.
     * 
     * This function processes the parsed TorchModel object and initializes
     * the model's internal structures and parameters. The implementation is specific
     * to each derived model class.
     * 
     * @param model The TorchModel object containing the model data.
     * @return true if parsing is successful, false otherwise.
     */
    virtual bool ParsingModel(oris_ai::TorchModel& model) = 0;

    /**
     * @brief Pure virtual function that performs a forward pass through the model.
     *
     * Each derived class (e.g., Yolov8) must implement this function to execute
     * the forward pass of the model.
     */
    virtual void Forward() = 0;

    /**
     * @brief Performs post-processing operations on model outputs.
     *
     * This function applies task-specific post-processing operations to the model outputs.
     * For detection tasks, it performs Non-Maximum Suppression (NMS) to filter overlapping
     * boxes. For segmentation tasks, it performs both NMS and mask coefficient filtering.
     * The goal is to process raw model outputs into meaningful detection or segmentation
     * results.
     *
     * The function is virtual and must be implemented by derived classes. The parameters
     * control the behavior of the post-processing operations, particularly the NMS
     * thresholds and maximum detection limit.
     *
     * @param score_threshold The confidence score threshold. Detections with scores below
     * this value will be discarded. Defaults to 0.25.
     * @param iou_threshold The IoU threshold for determining whether two boxes overlap
     * significantly. A lower value makes the suppression stricter. Defaults to 0.45.
     * @param max_det The maximum number of detections to retain after post-processing.
     * Defaults to 300.
     */
    virtual void PostProcess(float score_threshold = 0.25f,
                            float iou_threshold = 0.45f,
                            int max_det = 300) = 0;

    /**
     * @brief Retrieves the final detection results after processing.
     *
     * @return A constant reference to a vector of `Detection` objects, each representing a detected object.
     */
    virtual inline const std::vector<Detection>& GetDetectionResults() const = 0;

    // virtual inline const std::vector<float>& GetSegmentationMask() const = 0;

  protected:
    std::vector<std::unique_ptr<oris_ai::HiddenLayerAbstract<int8_t>>> layers_;  // Vector to manage all created layers

    std::unique_ptr<Tensor<int8_t>> input_image_tensor_;
    Device target_device_;

    size_t input_height_;
    size_t input_width_;
    float normalization_value_;

    uint8_t *gpu_input_data_;  // GPU memory pointer for input image data in NHWC format (OpenCV's default format)
    float first_quantization_scale_;
};

}  // namespace oris_ai
