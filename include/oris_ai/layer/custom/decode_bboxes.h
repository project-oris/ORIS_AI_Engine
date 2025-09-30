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

// #include "oris_ai/layer/concat.h"
#include "oris_ai/layer/split.h"

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
     * @brief Default constructor for the DecodeBboxes layer.
     * 
     * @param layer_name The name of the layer.
     * @param target_device The target device (CPU or GPU) on which the layer will run.
     */
    // DecodeBboxes(const std::string& layer_name, Device target_device) : HiddenLayerAbstract<T>
    // (layer_name), bboxes_split_size_(0) {}
    DecodeBboxes(const std::string& layer_name, Device target_device);

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
    std::unique_ptr<Split<T>> split_lr_rb_;
    // std::unique_ptr<Concat<T>> concat_x1y1_x2y2_;

    // std::vector<size_t> bboxes_split_size_; // Split size for bounding box distances
    // Tensor<T>* left_top_;             // Tensor to store left-top (lt) distance values
    // Tensor<T>* right_bottom_;         // Tensor to store right-bottom (rb) distance values
    Tensor<T>* x1y1_;
    Tensor<T>* x2y2_;
};

} // namespace oris_ai