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

#include "oris_ai/tensor/tensor.h"
#include "oris_ai/tensor/int8/tensor_int8.h"
#include "oris_ai_model.pb.h"

namespace oris_ai {

/**
 * @class LayerAbstract
 * @brief Represents a neural network layer with a specified name.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class LayerAbstract {
  public:
    /**
     * @brief Constructor to create a Layer object.
     * 
     * @param layer_name The name of the layer.
     */
    LayerAbstract(const std::string& layer_name);

    /**
     * @brief Virtual destructor for the LayerAbstract class.
     */
    virtual ~LayerAbstract() = default;

    /**
     * @brief Gets the input tensor at a specified index.
     * 
     * @param input_idx The index of the input tensor to retrieve. Default is 0.
     * @return A pointer to the input tensor at the specified index.
     */
    inline Tensor<T>* GetInputTensor(size_t input_idx = 0) {
      return input_tensors_.at(input_idx);
    }

    /**
     * @brief Sets the input tensor for the layer.
     * 
     * This is a virtual function that can be overridden by derived classes
     * to set the input tensor for the layer. The input tensor is typically used
     * as the data for forward propagation in the layer.
     * 
     * @param input_tensor A pointer to the input tensor.
     */
    virtual void SetInputTensor(Tensor<T>* input_tensor);

    /**
     * @brief Resets the input tensor at a specified index.
     * 
     * This function replaces the input tensor at the specified index with a new input tensor.
     * If the index is out of range, the function will throw an exception.
     * 
     * @param input_tensor A pointer to the new input tensor.
     * @param index The index of the input tensor to reset. Default is 0.
     */
    virtual void ResetInputTensor(Tensor<T>* input_tensor, size_t index = 0);

    /**
     * @brief Gets the number of input tensors.
     * 
     * @return The number of input tensors.
     */
    virtual inline size_t GetInputSize() { return input_tensors_.size(); }

    /**
     * @brief Pure virtual function to perform the forward pass of the layer.
     */
    virtual void Forward() = 0;

    /**
     * @brief Gets the name of the layer.
     * 
     * @return The actual name of the layer if built in Debug mode using CMake;
     *         otherwise, returns an empty string ("") in Release mode.
     */
    std::string GetLayerName() const;

    /**
     * @brief Sets the name of the layer.
     * 
     * @param layer_name The name of the layer.
     * 
     * @note This function operates correctly only when built in Debug mode using CMake.
     *       It does not operate in Release mode.
     */
    void SetLayerName(const std::string& layer_name);

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints input tensor.
     */
    void PrintInput(size_t index = 0);

    /**
     * @brief Prints a tensor.
     * 
     * @param tensor The tensor to print.
     * @param tensor_name The name of the tensor.
     */
    void PrintTensor(const Tensor<T>* tensor, const std::string& tensor_name);
#endif

  protected:
    std::vector<Tensor<T>*> input_tensors_;     // A vector of pointers to the input tensors
#ifdef USE_DEBUG_MODE
    std::string layer_name_;                    // The name of the layer
#endif
};

/**
 * @class HiddenLayerAbstract
 * @brief Represents a hidden layer with input and output tensors.
 * 
 * This class adds output tensor management to the base LayerAbstract class.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class HiddenLayerAbstract : public LayerAbstract<T> {
  public:
    /**
     * @brief Constructor to create a HiddenLayerAbstract object.
     */
    HiddenLayerAbstract(const std::string& layer_name) : LayerAbstract<T>(layer_name) {}

    /**
     * @brief Virtual destructor for the HiddenLayerAbstract class.
     */
    virtual ~HiddenLayerAbstract() = default;

    /**
     * @brief Gets the output tensor.
     * 
     * Retrieves the output tensor produced by the layer after forward propagation.
     * 
     * @return A pointer to the output tensor.
     */
    virtual inline Tensor<T>* GetOutputTensor() { return output_tensor_.get(); }

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints weight tensor (for debug).
     */
    virtual void PrintWeight() {}

    /**
     * @brief Prints output tensor (for debug).
     */
    virtual void PrintOutput(); 
#endif

  protected:
    std::unique_ptr<Tensor<T>> output_tensor_;  // A unique pointer to the output tensor
};

} // namespace oris_ai
