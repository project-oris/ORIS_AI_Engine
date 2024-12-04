/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"
#include "oris_ai/model/yolov8/yolov8_type.h"

#include <opencv2/opencv.hpp>

namespace oris_ai {

class Model {
  public:
    /**
     * @brief Constructor to initialize a Model with a given name.
     */
    Model(const std::string& model_name, Device target_device) : model_name_(model_name), target_device_(target_device), input_height_(0), input_width_(0), normalization_value_(1.0f) {}

    /**
     * @brief Virtual destructor for the Concat class.
     */
    virtual ~Model() = default;
    

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
     * @brief Converts and sets an input image as a tensor in NCHW format with normalization.
     *
     * This function processes an input image in OpenCV's `cv::Mat` format and prepares it for
     * use as a model input.
     * Normalize pixel values ​​by dividing each value by the normalization value specified in
     * CreateInputImageTensor(). The function transforms the image data from NHWC format (batch
     * size, height, width, channels) to the NCHW format (batch size, channels, height, width)
     * required by the model.
     *
     * The input image's shape must match the dimensions specified during the tensor creation;
     * otherwise, an error will be raised. The processed tensor data is subsequently
     * transferred to the target device (e.g., CPU or GPU).
     *
     * @param input_image The input image as a `cv::Mat` object. It must be a valid 3-channel
     * image matrix in BGR format by default.
     */
    void SetInputImageTensor(const cv::Mat& input_image);

    /**
     * @brief Pure virtual function to open the model.
     * 
     * Each derived class (e.g., Yolov8n) must implement this function.
     */
    virtual void Open(const std::string& model_path) = 0;

    /**
     * @brief Pure virtual function to open the model from an array.
     * 
     * This function allows the model to be loaded directly from a binary array
     * instead of a file. It is particularly useful when the model is embedded
     * as a resource within the application (e.g., via xxd).
     *
     * Each derived class (e.g., Yolov8n) must implement this function.
     *
     * @param model_data Pointer to the binary model data.
     * @param model_size Size of the binary model data in bytes.
     */
    virtual void OpenFromArray(const unsigned char* model_data, size_t model_size) = 0;

    /**
     * @brief Pure virtual function that performs a forward pass through the model.
     *
     * Each derived class (e.g., Yolov8) must implement this function to execute
     * the forward pass of the model.
     */
    virtual void Forward() = 0;

    /**
     * @brief Performs Non-Maximum Suppression (NMS) to filter overlapping detection boxes.
     *
     * This function applies Non-Maximum Suppression (NMS) to a set of detection boxes,
     * removing boxes with lower confidence scores that overlap significantly with
     * higher-confidence boxes.
     * The goal is to retain only the most relevant detections while eliminating redundant or 
     * overlapping boxes. 
     *
     * The function is virtual and must be implemented by derived classes. The parameters
     * control the behavior of the NMS operation, including the confidence score threshold,
     * Intersection over Union (IoU) threshold, and the maximum number of detections to keep.
     *
     * @param score_threshold The confidence score threshold. Detections with scores below
     * this value will be discarded. Defaults to 0.25.
     * @param iou_threshold The IoU threshold for determining whether two boxes overlap
     * significantly. A lower value makes the suppression stricter. Defaults to 0.45.
     * @param max_det The maximum number of detections to retain after applying NMS. Defaults
     * to 300.
     */
    virtual void NonMaxSuppression(float score_threshold = 0.25f, float iou_threshold = 0.45f, int max_det = 300) = 0;

    /**
     * @brief Retrieves the final detection results after processing.
     *
     * @return A constant reference to a vector of `Detection` objects, each representing a detected object.
     */
    virtual inline const std::vector<Detection>& GetResult() const = 0;


  protected:
    std::string model_name_;  // The name of the model
    std::vector<std::unique_ptr<oris_ai::HiddenLayerAbstract<float>>> layers_;  // Vector to manage all created layers

    std::unique_ptr<Tensor<float>> input_image_tensor_;
    Device target_device_;

    size_t input_height_;
    size_t input_width_;
    float normalization_value_;
};

/**
 * @brief Factory function to create a model based on the model name.
 */
std::unique_ptr<Model> CreateModel(const std::string& model_name, Device target_device);

}  // namespace oris_ai
