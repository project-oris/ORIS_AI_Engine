/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/common/tensor.h"
#include "oris_ai/protobuf/oris_ai_yolov8.pb.h"

namespace oris_ai {

/**
 * @enum LayerType
 * @brief Defines types of layers that can be used in the neural network.
 */
enum class LayerType {
  CONV,
  // BATCHNORM,
  ACTIVATION,
  MAXPOOL,
  CONCAT,
  UPSAMPLE   
};

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
     * @brief Constructor to create a Layer object with a given name.
     * 
     * @param layer_name The name of the layer.
     */
    LayerAbstract(const std::string& name) : name_(name) {}

    /**
     * @brief Virtual destructor for the LayerAbstract class.
     */
    virtual ~LayerAbstract() {}

    /**
     * @brief Sets the input tensor for the layer.
     * 
     * @param input_tensor A pointer to the input tensor.
     */
    void SetInputTensor(Tensor<T>* input_tensor);

    /**
     * @brief Gets the input tensor at a specified index.
     * 
     * @param input_idx The index of the input tensor to retrieve. Default is 0.
     * @return A pointer to the input tensor at the specified index.
     */
    inline Tensor<T>* GetInputTensor(unsigned int input_idx = 0) {
      return input_tensors_.at(input_idx);
    }

    /**
     * @brief Gets the number of input tensors.
     * 
     * @return The number of input tensors.
     */
    inline unsigned int GetInputSize() { return input_tensors_.size(); }

    /**
     * @brief Gets the name of the layer.
     * 
     * @return The name of the layer.
     */
    inline const std::string& GetLayerName() const { return name_; }

    /**
     * @brief Gets the output tensor.
     * 
     * @return A pointer to the output tensor.
     */
    inline Tensor<T>* GetOutputTensor() { return output_tensor_.get(); }

    /**
     * @brief Pure virtual function to perform the forward pass of the layer.
     */
    virtual void Forward() = 0;

#ifdef USE_DEBUG_MODE
    /**
     * @brief Prints input tensor (for debug).
     */
    void PrintInput(unsigned int index = 0);

    /**
     * @brief Prints output tensor (for debug).
     */
    void PrintOutput();

    /**
     * @brief Prints weight tensor (for debug).
     */
    virtual void PrintWeight() {}
#endif

  protected:
    std::string name_;                    /** The name of the layer */
    std::vector<Tensor<T>*> input_tensors_;     /** A vector of pointers to the input tensors */
    std::unique_ptr<Tensor<T>> output_tensor_;  /** A unique pointer to the output tensor */
};

} // namespace oris_ai
