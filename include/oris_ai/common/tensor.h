/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include <cstddef>  // for size_t
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
     * @brief Retrieves a pointer to the data stored in CPU memory, copying from GPU if necessary.
     * 
     * @return T* Pointer to the data in CPU memory.
     */
    T* GetCPUDataPtr();

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
     * specified device. If the data is already on the specified device, no transfer is performed.
     * 
     * @param device The target device to transfer the data to. 
     *               Use Device::CPU to move data to CPU memory or Device::GPU to move data to GPU memory.
     *
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
     * @brief Permute the dimensions of the tensor.
     * 
     * @param dims A vector specifying the new order of dimensions.
     */
    void Permute(const std::vector<size_t>& dims);

  private:
    std::vector<size_t> shape_;     /** Stores the shape (dimensions) of the tensor */
    size_t total_count_;            /** Total number of elements in the tensor */
    size_t total_bytes_;            /** Total number of bytes required by the tensor */
    Device head_;                   /** Tracks the device with the latest valid data */

    T* cpu_data_ptr_;               /** Pointer to the data stored in CPU memory */
    T* gpu_data_ptr_;               /** Pointer to the data stored in GPU memory */
    bool cpu_only_;                 /** Determines whether to use only CPU */

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
     * @param indices A pointer to an array of size_t representing the multi-dimensional indices.
     * @return The flat index corresponding to the given multi-dimensional indices.
     */
    size_t FlattenIndex(const size_t* indices) const;

    /**
     * @brief Converts multi-dimensional indices to a flat index for 1D memory access, using a specified shape.
     * 
     * This function calculates the flat index corresponding to a given multi-dimensional index based on the 
     * provided shape of the tensor. This is useful for accessing elements in a linearized 1D memory layout.
     * 
     * @param indices A vector of indices specifying the location in the tensor.
     * @param shape A vector representing the shape (dimensions) of the tensor to use for flattening.
     * @return The flat index corresponding to the given multi-dimensional indices.
     */
    // size_t FlattenIndex(const std::vector<size_t>& indices, const std::vector<size_t>& shape) const;

    /**
     * @brief Increments multi-dimensional indices for tensor traversal.
     * 
     * This function takes a multi-dimensional index and increments it to point to the next element
     * in the tensor, based on the specified shape. If the index reaches the end of a given dimension,
     * it resets that dimension's index to zero and carries over the increment to the next more significant dimension.
     * 
     * @param indices A vector of indices to be incremented.
     * @param shape A vector representing the shape (dimensions) of the tensor.
     * @return A boolean value indicating whether the increment was successful.
     *         Returns false if the end of all dimensions has been reached.
     */
    bool IncrementIndices(std::vector<size_t>& indices, const std::vector<size_t>& shape) const;
};

} // namespace oris_ai
