/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/common/tensor.h"
#include "oris_ai/protobuf/oris_ai_yolov8.pb.h"

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
     * @brief Constructor to create a Layer object without layer_name.
     */
    LayerAbstract() : layer_name_("") {}

    /**
     * @brief Constructor to create a Layer object with a given layer_name.
     * 
     * @param layer_name The name of the layer.
     */
    LayerAbstract(const std::string& layer_name) : layer_name_(layer_name) {}

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
     * @brief Gets the number of input tensors.
     * 
     * @return The number of input tensors.
     */
    virtual inline size_t GetInputSize() { return input_tensors_.size(); }

    /**
     * @brief Gets the name of the layer.
     * 
     * @return The name of the layer.
     */
    inline const std::string& GetLayerName() const { return layer_name_; }

    inline void SetLayerName(const std::string& layer_name) { layer_name_ = layer_name; }

    /**
     * @brief Pure virtual function to perform the forward pass of the layer.
     */
    virtual void Forward() = 0;

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints input tensor (for debug).
     */
    void PrintInput(size_t index = 0);
#endif

  protected:
    std::string layer_name_;                    // The name of the layer
    std::vector<Tensor<T>*> input_tensors_;     // A vector of pointers to the input tensors
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
     * @brief Constructor to create a HiddenLayerAbstract object without layer_name.
     */
    HiddenLayerAbstract() : LayerAbstract<T>() {}

    /**
     * @brief Constructor to create a HiddenLayerAbstract object with layer_name.
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
