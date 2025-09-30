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

namespace oris_ai {

/**
 * @class Tensor
 * @brief A class for managing multi-dimensional arrays (tensors) that can be stored in either CPU or GPU memory.
 */
template <>
class Tensor<int8_t> {
  public:
    /**
     * @brief Constructor for creating a Tensor object.
     * 
     * @param shape Shape of the tensor.
     * @param cpu_only If true, the tensor will only allocate memory on the CPU. Default is false.
     */
    Tensor(const std::vector<size_t>& shape, bool cpu_only = false);

    /**
     * @brief Destructor that frees the allocated memory.
     */
    ~Tensor();

    /**
     * @brief Retrieves a pointer to the data stored in CPU memory, copying from GPU if
     * necessary.
     * 
     * @return int8_t* Pointer to the data in CPU memory.
     */
    int8_t* GetCPUDataPtr();

    /**
     * @brief Retrieves a constant pointer to the data stored in CPU memory, copying from GPU
     * if necessary. Optionally retains the current device state as GPU.
     * 
     * @param keep_head_gpu If true, retains the current device state as GPU even after
     * copying data to CPU.
     * @return const int8_t* A constant pointer to the data in CPU memory.
     */
    const int8_t* GetConstCPUDataPtr(bool keep_head_gpu = false) const;

    /**
     * @brief Retrieves a pointer to the data stored in GPU memory, copying from CPU if
     * necessary.
     * 
     * @return int8_t* Pointer to the data in GPU memory.
     */
    int8_t* GetGPUDataPtr();

    /**
     * @brief Retrieves a constant pointer to the data stored in GPU memory, copying from CPU
     * if necessary.
     * 
     * @return const int8_t* A constant pointer to the data in GPU memory.
     */
    const int8_t* GetConstGPUDataPtr() const;

    /**
     * @brief Retrieves a reference to the tensor element in CPU memory at a specific multi-dimensional index, copying from GPU if necessary.
     * 
     * @param indices A vector specifying the multi-dimensional index of the desired element in the tensor.
     * @return int8_t& Reference to the element in CPU memory at the specified index.
     */
    int8_t& GetCPUData(const std::vector<size_t>& indices);

    /**
     * @brief Set the Tensor data to a specific value for all elements.
     * 
     * @param init The value to initialize the data with.
     */
    void SetCPUData(int8_t init);

    /**
     * @brief Transfers the tensor data to the specified device (CPU or GPU).
     * This function transfers the tensor's data between CPU and GPU, depending on the
     * specified device. If the data is already on the specified device, no transfer is
     * performed.
     * 
     * @param device The target device to transfer the data to. Use Device::CPU to move data
     * to CPU memory or Device::GPU to move data to GPU memory.
     */
    void To(Device device);

    /**
     * @brief Returns the current device where the tensor data is stored.
     * 
     * This function returns the device (CPU or GPU) that currently holds the valid data
     * for this tensor. The returned value indicates where the most recent valid data
     * is located.
     * 
     * @return Device The current device where the tensor data is stored.
     */
    inline Device GetDevice() { return head_; }

    /**
     * @brief Returns the shape of the tensor.
     * 
     * This function returns the dimensions of the tensor as a vector of size_t.
     * 
     * @return A vector representing the shape (dimensions) of the tensor.
     */
    inline const std::vector<size_t>& GetShape() const { return shape_; }

    /** 
     * @brief Returns the total count of the tensor.
     */
    inline size_t GetTotalCount() const { return total_count_; }

    /**
     * @brief Sets the view shape of the tensor.
     * 
     * The view shape allows the tensor to have a temporary shape different from its original
     * shape without altering the actual data storage. It enables reshaping operations that
     * are non-destructive.
     * 
     * @param view_shape The new shape to be set as the view shape.
     */
    inline void SetViewShape(const std::vector<size_t>& view_shape) { view_shape_ = view_shape; }

    /**
     * @brief Retrieves the view shape of the tensor if set.
     * 
     * Returns the current view shape, if one has been specified, allowing temporary reshaping
     * of the tensor without altering its actual data layout.
     * 
     * @return const std::vector<size_t>& The view shape of the tensor.
     */
    inline const std::vector<size_t>& GetViewShape() const { return view_shape_; }

    /**
     * @brief Transposes two specified dimensions of the tensor.
     * 
     * Swaps the data along the specified dimensions (dim1 and dim2) for transposing. 
     * Optionally allows using the view shape instead of the original shape, depending on the 
     * `use_view_input` parameter.
     * 
     * @param dim1 The first dimension to transpose.
     * @param dim2 The second dimension to transpose.
     * @param use_view_shape If true, applies the transpose to the view shape if set;
     * otherwise, applies to the main shape.
     */
    void Transpose(size_t dim1, size_t dim2, bool use_view_shape = false);

    /**
     * @brief Calculates the stride for a specific dimension in the tensor.
     * 
     * This function computes the stride for a given dimension, which represents
     * the number of elements to skip in memory to move to the next element along that
     * dimension. The stride is calculated based on the dimensions of the tensor shape.
     * 
     * @param dim The dimension index for which to calculate the stride.
     * @param use_view_shape If true, applies the transpose to the view shape if set;
     * otherwise, applies to the main shape.
     * @return The calculated stride for the specified dimension.
     */
    size_t GetDimensionStride(size_t dim, bool use_view_shape = false) const;

#ifdef USE_TENSOR_EXTRA_FUNCTIONS
    // sthong note
    //  - The Add() function is currently commented out as it is rarely used in the Tensor class.
    //  - Tensor addition operations are frequently performed at the layer level, and it is recommended
    //    to use the Elementwise layer for tensor addition operations at the layer level.
    /**
     * @brief Adds the values of another Tensor to this Tensor.
     * 
     * This method adds the values of the input tensor to the current tensor.
     * Both tensors must have the same shape.
     * 
     * @param other The other tensor whose values will be added to this tensor.
     */
    // void Add(const Tensor<T>& other);

    /**
     * @brief Permute the dimensions of the tensor.
     * 
     * @param dims A vector specifying the new order of dimensions.
     */
    void Permute(const std::vector<size_t>& dims);

    /**
     * @brief Splits the current tensor into multiple smaller tensors along the specified dimension.
     * 
     * This function divides the current tensor into a set of smaller tensors along a
     * specified dimension. The split sizes are provided as a list, and each resulting tensor
     * will have the same shape as the original tensor, except for the specified dimension,
     * where the size  will be divided according to `split_sizes`.
     * The split tensors are provided as an `std::initializer_list` to minimize memory
     * overhead and optimize performance for cases where the split configuration is fixed and
     * does not require dynamic resizing.
     *
     * @param split_sizes A vector specifying the sizes of each split segment along the split
     * dimension.
     * @param split_dim The dimension along which the tensor will be split.
     * @param split_tensors An initializer list containing pointers to tensors that will hold
     * the results of the split.
     * @param use_view_input If true, applies the Split to the view shape if set;
     * otherwise, applies to the main shape.
     */
    void Split(const std::vector<size_t>& split_sizes, size_t split_dim, const std::initializer_list<Tensor<T>*>& split_tensors, bool use_view_input=false);
#endif

#ifdef USE_TENSOR_OPERATOR_OVERLOADING
    /**
     * @brief Overloads the + operator for element-wise addition of two Tensors.
     * 
     * Creates and returns a new Tensor, where each element is the sum of the corresponding
     * elements in the current Tensor and the provided Tensor `other`.
     * 
     * @param other The Tensor to be added element-wise to the current Tensor.
     * @return Tensor A new Tensor containing the result of the addition.
     */
    Tensor operator+(const Tensor& other) const;

    /**
     * @brief Overloads the - operator for element-wise subtraction of two Tensors.
     * 
     * Creates and returns a new Tensor, where each element is the result of subtracting 
     * the corresponding element in the provided Tensor `other` from the current Tensor.
     * 
     * @param other The Tensor to be subtracted element-wise from the current Tensor.
     * @return Tensor A new Tensor containing the result of the subtraction.
     */
    Tensor operator-(const Tensor& other) const;

    /**
     * @brief Overloads the * operator for element-wise scalar multiplication of a Tensor.
     * 
     * Creates and returns a new Tensor, where each element in the current Tensor is multiplied
     * by the provided scalar value. The result is a new Tensor with the same shape as the original,
     * containing the scaled values.
     * 
     * @param scalar The scalar value to multiply each element of the Tensor by.
     * @return Tensor A new Tensor containing the result of the scalar multiplication.
     */
    Tensor operator*(const int8_t scalar) const;

    /**
     * @brief Overloads the += operator for in-place element-wise addition of two Tensors.
     * 
     * Adds the values of the provided Tensor `other` to the current Tensor in-place.
     * Both tensors must have the same shape.
     * 
     * @param other The Tensor whose values will be added to the current Tensor.
     * @return Tensor& Reference to the current Tensor after addition.
     */
    Tensor<int8_t>& operator+=(const Tensor& other);

    /**
     * @brief Overloads the *= operator for in-place element-wise scalar multiplication.
     * 
     * Multiplies each element of the current Tensor by the provided scalar value in-place.
     * 
     * @param scalar The scalar value to multiply each element of the Tensor by.
     * @return Tensor& Reference to the current Tensor after multiplication.
     */
    Tensor<int8_t>& operator*=(const int8_t scalar);
#endif

    inline float GetOutputScale() const { return output_scale_; }
    inline void SetOutputScale(float output_scale) { output_scale_ = output_scale; }

  private:
    std::vector<size_t> shape_;     // Stores the shape (dimensions) of the tensor
    std::vector<size_t> view_shape_;// Stores the view shape (dimensions) of the tensor
    size_t total_count_;            // Total number of elements in the tensor
    size_t total_bytes_;            // Total number of bytes required by the tensor
    Device head_;                   // Tracks the device with the latest valid data

    int8_t* cpu_data_ptr_;          // Pointer to the data stored in CPU memory
    int8_t* gpu_data_ptr_;          // Pointer to the data stored in GPU memory
    bool cpu_only_;                 // Determines whether to use only CPU
    float output_scale_;            // Output scale of the tensor

    /**
     * @brief Allocates memory for the tensor in CPU memory.
     */
    void AllocateCPUMemory();

    /**
     * @brief Allocates memory for the tensor in GPU memory.
     */
    void AllocateGPUMemory();

    /**
     * @brief Ensures the tensor data is ready in CPU memory, copying from GPU if necessary.
     * This method consolidates the common logic for transferring data from GPU
     * to CPU or initializing CPU state when the head is uninitialized.
     * 
     * @param keep_head_gpu If true, retains the current device state as GPU even after
     * copying data to CPU. This is useful when GPU state needs to be preserved
     * for further operations without switching the head to CPU.
     */
    void EnsureCPUDataReady(bool keep_head_gpu = false);

    /**
     * @brief Handles the copying of data from CPU to GPU and updates the head.
     * This method consolidates the common logic for transferring data from CPU
     * to GPU or initializing GPU state when the head is uninitialized.
     */
    void EnsureGPUDataReady();

    /**
     * @brief Converts multi-dimensional indices to a flat index for 1D memory access.
     * @param indices A pointer to an array of size_t representing the multi-dimensional
     * indices.
     * @return The flat index corresponding to the given multi-dimensional indices.
     */
    size_t FlattenIndex(const size_t* indices) const;

    /**
     * @brief Increments multi-dimensional indices for tensor traversal.
     * 
     * This function takes a pointer to a multi-dimensional index array and increments it to
     * point to the next element in the tensor, based on the specified shape. If the index
     * reaches the end of a given dimension, it resets that dimension's index to zero and
     * carries over the increment to the next more significant dimension.
     *  
     * @param indices A pointer to an array of indices to be incremented.
     * @param shape A vector representing the shape (dimensions) of the tensor.
     * @return A boolean value indicating whether the increment was successful.
     *         Returns false if the end of all dimensions has been reached.
     */
    bool IncrementIndices(size_t* indices, const std::vector<size_t>& shape) const;
};

} // namespace oris_ai
