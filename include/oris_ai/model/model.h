/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

#include <opencv2/opencv.hpp>

namespace oris_ai {

class Model {
  public:
    /**
     * @brief Constructor to initialize a Model with a given name.
     */
    Model(const std::string& model_name, Device target_device, size_t input_height, size_t input_width) : model_name_(model_name), target_device_(target_device), input_height_(input_height), input_width_(input_width) {}

    /**
     * @brief Virtual destructor for the Concat class.
     */
    virtual ~Model() = default;

    /**
     * @brief Sets the input image as a tensor for the model, with optional RGB swapping and normalization.
     *
     * This function converts an input image (in OpenCV's `cv::Mat` format) to a tensor
     * that can be used as the input for the model. It supports optional BGR to RGB 
     * conversion and allows normalization of the pixel values.
     *
     * The input image is assumed to be in BGR format by default, which is common for 
     * images loaded using OpenCV. If the `rb_swap` flag is set to `true`, the function 
     * will convert the image from BGR to RGB. After optional conversion, the pixel values 
     * of the image will be normalized by dividing each value by the given `normalization_value`.
     *
     * The image will be transformed from OpenCV's default NHWC format (batch size, height, width, channels) to the NCHW format required by the model.
     *
     * @param input_image The input image in `cv::Mat` format. It should be a valid image matrix.
     * @param normalization_value The value to normalize pixel values. Typically, it is 255.0 for images with pixel values in the 0-255 range.
     * @param rb_swap If set to `true`, the function will convert the image from BGR to RGB format. Defaults to `false` (BGR format).
     */
    void SetInputImageTensor(const cv::Mat& input_image, float normalization_value, bool rb_swap = false);

    /**
     * @brief Pure virtual function to open the model.
     * Each derived class (e.g., Yolov8n) must implement this function.
     */
    virtual bool Open(const std::string& model_path) = 0;

    /**
     * @brief Pure virtual function that performs a forward pass through the model.
     *
     * Each derived class (e.g., Yolov8) must implement this function to execute
     * the forward pass of the model.
     */
    virtual void Forward() = 0;

  protected:
    void CreateDummyInputTensor();

    std::string model_name_;  // The name of the model
    std::vector<std::unique_ptr<oris_ai::HiddenLayerAbstract<float>>> layers_;  // Vector to manage all created layers

    std::unique_ptr<Tensor<float>> input_image_tensor_;
    Device target_device_;

    size_t input_height_;
    size_t input_width_;
};

/**
 * @brief Factory function to create a model based on the model name.
 */
std::unique_ptr<Model> CreateModel(const std::string& model_name, Device target_device);

}  // namespace oris_ai
