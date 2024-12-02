/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

// #include "oris_ai/layer/concat.h"
#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class DecodeBboxes
 * @brief Represents a bounding box decoding layer.
 * 
 * This class is designed to decode bounding boxes from predicted offsets and anchor points.
 * It transforms distance offsets to bounding boxes in xywh format.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class DecodeBboxes : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Default constructor for the DecodeBboxes layer with layer_name.
     * 
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     */
    DecodeBboxes(const std::string& layer_name, Device target_device) : HiddenLayerAbstract<T>(layer_name), bboxes_split_size_(0) {}

    /**
     * @brief Destructor for the DecodeBboxes class.
     */
    ~DecodeBboxes();

    /**
     * @brief Initializes the DecodeBboxes layer.
     */
    void InitDecodeBboxes();

    /**
     * @brief Pure virtual function to perform the forward pass of the DecodeBboxes layer.
     * 
     * This function must be implemented by derived classes, such as DecodeBboxesCPU
     * or DecodeBboxesGPU, which are specific to the execution device (e.g., CPU or GPU). 
     */
    virtual void Forward() = 0;

  protected:
    // std::unique_ptr<Concat<T>> concat_c_xy_wh_;

    std::vector<size_t> bboxes_split_size_; // Split size for bounding box distances
    Tensor<T>* left_top_;             // Tensor to store left-top (lt) distance values
    Tensor<T>* right_bottom_;         // Tensor to store right-bottom (rb) distance values
};

} // namespace oris_ai