/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include <cstddef>  // for size_t
#include <memory>
#include <vector>

namespace oris_ai {

/**
 * @enum Device
 * @brief Enum to specify the device where the tensor is stored (UNINIT, CPU, GPU).
 */
enum class Device { 
  UNINIT, /** Tensor location is not initialized */
  CPU,    /** Tensor stored in CPU memory */
  GPU     /** Tensor stored in GPU memory */
};

/**
 * @class Tensor
 * @brief A class for managing multi-dimensional arrays (tensors) that can be stored in either CPU or GPU memory.
 * 
 * @tparam T The data type of the tensor elements (e.g., float).
 */
template <typename T>
class Tensor {
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
    Tensor operator*(const T scalar) const;

    Tensor<T>& operator+=(const Tensor& other);

    Tensor<T>& operator*=(const T scalar);

    /**
     * @brief Retrieves a pointer to the data stored in CPU memory, copying from GPU if
     * necessary.
     * 
     * @return T* Pointer to the data in CPU memory.
     */
    T* GetCPUDataPtr();

    /**
     * @brief Retrieves a constant pointer to the data stored in CPU memory, copying from GPU
     * if necessary.
     * 
     * @return const T* A constant pointer to the data in CPU memory.
     */
    const T* GetCPUDataPtr() const;

    /**
     * @brief Retrieves a pointer to the data stored in GPU memory, copying from CPU if necessary.
     * 
     * @return T* Pointer to the data in GPU memory.
     */
    T* GetGPUDataPtr();

    /**
     * @brief Retrieves a reference to the tensor element in CPU memory at a specific multi-dimensional index, copying from GPU if necessary.
     * 
     * @param indices A vector specifying the multi-dimensional index of the desired element in the tensor.
     * @return T& Reference to the element in CPU memory at the specified index.
     */
    T& GetCPUData(const std::vector<size_t>& indices);

    /**
     * @brief Set the Tensor data to a specific value for all elements.
     * 
     * @param init The value to initialize the data with.
     */
    void SetCPUData(T init);

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
    inline const size_t GetTotalCount() const { return total_count_; }

    /**
     * @brief Adds the values of another Tensor to this Tensor.
     * 
     * This method adds the values of the input tensor to the current tensor.
     * Both tensors must have the same shape.
     * 
     * @param other The other tensor whose values will be added to this tensor.
     */
    void Add(const Tensor<T>& other);

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

  private:
    std::vector<size_t> shape_;     // Stores the shape (dimensions) of the tensor
    std::vector<size_t> view_shape_;// Stores the view shape (dimensions) of the tensor
    size_t total_count_;            // Total number of elements in the tensor
    size_t total_bytes_;            // Total number of bytes required by the tensor
    Device head_;                   // Tracks the device with the latest valid data

    T* cpu_data_ptr_;               // Pointer to the data stored in CPU memory
    T* gpu_data_ptr_;               // Pointer to the data stored in GPU memory
    bool cpu_only_;                 // Determines whether to use only CPU

    /**
     * @brief Allocates memory for the tensor in CPU memory.
     */
    void AllocateCPUMemory();

    /**
     * @brief Allocates memory for the tensor in GPU memory.
     */
    void AllocateGPUMemory();

    /**
     * @brief Handles the copying of data from GPU to CPU and updates the head.
     * This method consolidates the common logic for transferring data from GPU
     * to CPU or initializing CPU state when the head is uninitialized.
     */
    void EnsureCPUDataReady();

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
};

} // namespace oris_ai
