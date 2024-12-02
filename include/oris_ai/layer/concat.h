/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/layer.h"

namespace oris_ai {

/**
 * @class Concat
 * @brief Represents a concatenation layer in a neural network.
 * 
 * This class defines a concatenation layer, which concatenates multiple input tensors 
 * along a specified dimension. The Concat class can optionally apply a Permute operation 
 * to adjust the shape of each input tensor before concatenation.
 * 
 * @tparam T The data type for the layer operations (e.g., float).
 */
template <typename T>
class Concat : public HiddenLayerAbstract<T> {
  public:
    /**
     * @brief Constructor to initialize a concatenation layer without layer_name.
     */
    Concat() : HiddenLayerAbstract<T>(), concat_dim_(1), use_view_input_(false) {}

    /**
     * @brief Constructor to initialize a concatenation layer.
     * @param name The name of the layer.
     */
    Concat(const std::string& layer_name) 
      : HiddenLayerAbstract<T>(layer_name), concat_dim_(1), use_view_input_(false) {}

    /**
     * @brief Destructor for the Concat class.
     */
    // ~Concat();
    ~Concat() = default;

    /**
     * @brief Initializes the concat layer with parameters from a TorchConcat object.
     * 
     * @param concat_params The TorchConcat object containing concat parameters.
     */
    void InitConcat(const TorchConcat& concat_params);

    /**
     * @brief Initializes the concat layer with the specified concatenation dimension.
     * @param concat_dim The dimension along which the tensors are concatenated.
     */
    void InitConcat(const size_t concat_dim);

    /**
     * @brief Pure virtual function to perform the forward pass of the concat layer.
     * 
     * This function must be implemented by derived classes, which are specific
     * to the execution device (e.g., CPU or GPU). It performs the concat operation.
     */
    virtual void Forward() = 0;

    /**
     * @brief Enables the use of the view input mode for the Concat layer.
     * 
     * This function sets the internal flag to indicate that the view input mode should be
     * used. When enabled, operations in the Concat layer may utilize a view of the input
     * tensor rather than directly accessing or modifying the original data.
     */
    inline void SetUseViewInput() { use_view_input_ = true; }

  protected:
    /**
     * @brief Configures and initializes the output tensor for the concat layer.
     * @note This function is automatically called by InitConcat() after setting the 
     * concatenation dimension.
     */
    void Setup();

    size_t concat_dim_; // The dimension along which concatenation occurs
    std::vector<size_t> output_shape_; // Shape of the output tensor after concatenation

    bool use_view_input_; // Flag indicating whether to use view input mode for tensor operations
};

} // namespace oris_ai
