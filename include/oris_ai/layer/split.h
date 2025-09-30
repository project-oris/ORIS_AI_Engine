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

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class Split
 * @brief Represents a split layer in a neural network.
 * 
 * This class defines a split layer that divides the input tensor along a specified dimension 
 * into multiple smaller tensors. Each split tensor contains a specified size along the split
 * dimension. This allows for flexible data handling and supports efficient tensor
 * partitioning across different parts of the network.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Split : public LayerAbstract<T> {
  public:
    /**
     * @brief Default constructor to initialize the Split layer.
     * @param layer_name The name of the layer.
     */
    Split(const std::string& layer_name)
      : LayerAbstract<T>(layer_name),
        split_dim_(0),
        use_view_input_shape_(false) {}

    /**
     * @brief Default destructor for the Split class.
     */
    ~Split() = default;

    /**
     * @brief Initializes the Split layer by setting up the dimension to split along and
     * preparing output tensors for each split segment.
     * 
     * This function configures the Split layer by specifying the split dimension and the sizes
     * for each split section. It prepares the output tensors that will store the partitioned
     * data segments along the specified dimension.
     * 
     * @param split_sizes A vector containing the sizes for each split along split_dim. 
     *                    Each element in the vector represents the size of a split section.
     * @param split_dim The dimension along which to split the input tensor.
     * @param use_view_input_shape Flag indicating whether to use the view shape of the input tensor.
     */
    void InitSplit(const std::vector<size_t>& split_sizes, size_t split_dim, bool use_view_input_shape = false);

    /**
     * @brief Returns a pointer to the split tensor at the specified index.
     * 
     * This function allows access to individual split tensors that result from the 
     * partitioning of the input tensor. It retrieves the tensor at the given index.
     * 
     * @param index The index of the desired split tensor.
     * @return A pointer to the requested split tensor.
     */
    inline Tensor<T>* GetSplitTensor(size_t index) const { return split_tensors_.at(index).get(); }

    /**
     * @brief Pure virtual function to perform the forward pass of the split layer.
     * 
     * This function is implemented in derived classes for specific execution devices 
     * (e.g., CPU or GPU). It performs the actual tensor splitting operation by copying
     * data from the input tensor into each of the split tensors according to the split sizes.
     */
    virtual void Forward() = 0;

  protected:
    size_t split_dim_;                          // Dimension along which to split the input tensor.
    std::vector<size_t> split_sizes_;           // Sizes for each split along split_dim.
    std::vector<std::unique_ptr<Tensor<T>>> split_tensors_; // Unique pointers to the resulting split tensors.
    size_t stride_;                             // Stride size along the split dimension for efficient data access.
    bool use_view_input_shape_;                 // Flag to indicate whether view shape should be used.
};

} // namespace oris_ai
