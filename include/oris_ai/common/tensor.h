/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#ifndef INCLUDE_ORIS_AI_COMMON_TENSOR_H_
#define INCLUDE_ORIS_AI_COMMON_TENSOR_H_

#include <vector>

namespace oris_ai {

  /**
   * @enum Device
   * @brief Enum to specify the device where the tensor is stored (CPU, GPU, SYNCED).
   */
  enum class Device { 
      CPU,   /** Tensor stored in CPU memory */
      GPU,   /** Tensor stored in GPU memory */
      SYNCED /** Tensor data is synchronized across CPU and GPU */
  };

  /**
   * @class Tensor
   * @brief A class for managing multi-dimensional arrays (tensors) that can be stored in either CPU or GPU memory.
   * 
   * @tparam T The data type of the tensor elements (e.g., float, int).
   */
  template <typename T>
  class Tensor {
  public:
    /**
     * @brief Constructs a Tensor with the given shape and device.
     * 
     * This constructor initializes a Tensor object with the specified shape and allocates memory 
     * on the specified device. 
     * 
     * - If the device is set to `Device::CPU`, the tensor will be allocated and used only in CPU memory.
     * - If the device is set to `Device::GPU`, the tensor will be allocated in both CPU and GPU memory, 
     *   allowing data to be transferred and used across both devices.
     * 
     * @param shape A vector specifying the dimensions of the tensor.
     * @param device The device where the tensor will be allocated. The default is `Device::CPU`.
     */
    Tensor(const std::vector<size_t>& shape, Device device = Device::CPU);


    /**
     * @brief Destructor that frees the allocated memory.
     */
    ~Tensor();

    /**
     * @brief Retrieves the data from CPU memory, copying from GPU if necessary.
     * 
     * This method provides two functionalities based on the parameters passed:
     * - If no parameters are provided, it returns a pointer to the entire data stored in CPU memory.
     * - If a multi-dimensional index is provided, it returns a reference to the specific element at the given index.
     * 
     * @return T* Pointer to the data in CPU memory.
     */
    T* GetCPUData();

    /**
     * @brief Retrieves the element from CPU memory at the specified multi-dimensional index, copying from GPU if necessary.
     * 
     * This method takes a vector of indices representing the multi-dimensional index in the tensor.
     * It converts the multi-dimensional index into a 1D index and returns a reference to the element at that position.
     * If the data is currently stored on the GPU, it is copied to the CPU before accessing the element.
     * 
     * @param indices A vector specifying the multi-dimensional index of the element.
     * @return T& Reference to the element in CPU memory at the specified index.
     * @throws std::invalid_argument if the number of indices does not match the tensor dimensions.
     * 
     * @code
     * // Example usage:
     * // Suppose we have a 3D tensor with dimensions 4x4x4.
     * std::vector<size_t> shape = {4, 4, 4};
     * oris_ai::Tensor<float> tensor(shape, oris_ai::Device::CPU);
     * 
     * // Set some data in the tensor (for example, initializing the tensor with 1.0).
     * tensor.SetCPUData(1.0f);
     * 
     * // Now, retrieve the element at the position (2, 3, 1).
     * std::vector<size_t> indices = {2, 3, 1};
     * float& value = tensor.GetCPUData(indices);
     * 
     * // You can now access or modify the element at this position.
     * std::cout << "The value at (2, 3, 1) is: " << value << std::endl;
     * 
     * // Modify the value at (2, 3, 1).
     * value = 5.0f;
     * 
     * // Verify the change.
     * std::cout << "The modified value at (2, 3, 1) is: " << tensor.GetCPUData(indices) << std::endl;
     * @endcode
     */
    T& GetCPUData(const std::vector<size_t>& indices);



    /**
     * @brief Retrieves the data from GPU memory, copying from CPU if necessary.
     * 
     * @return Pointer to the data in GPU memory.
     */
    T* GetGPUData();

    /**
     * @brief Initializes the tensor with a specific value.
     * 
     * @param init The value to initialize the data with.
     */
    void SetCPUData(T init);

    /**
     * @brief Transfers the tensor data to the specified device (CPU or GPU).
     *
     * This function transfers the tensor's data between CPU and GPU, depending on the
     * specified device. If the data is already on the specified device, no transfer is performed.
     * 
     * @param device The target device to transfer the data to. 
     *               Use Device::CPU to move data to CPU memory or Device::GPU to move data to GPU memory.
     *
     * @note This function internally calls either GetCPUData() or GetGPUData() to handle
     *       the memory transfer.
     * 
     * @example
     * @code
     * oris_ai::Tensor<float> tensor({4, 4, 4}, oris_ai::Device::CPU);
     * tensor.SetCPUData(1.0f);
     * 
     * // Transfer data to GPU
     * tensor.To(oris_ai::Device::GPU);
     * 
     * // Transfer data back to CPU
     * tensor.To(oris_ai::Device::CPU);
     * @endcode
     */
    void To(Device device);

    /**
     * @brief Returns the shape of the tensor.
     * 
     * This function returns the dimensions of the tensor as a vector of size_t.
     * 
     * @return A vector representing the shape (dimensions) of the tensor.
     */
    const std::vector<size_t>& Shape() const;

  private:
    std::vector<size_t> shape_;     /** Stores the shape (dimensions) of the tensor */
    size_t total_count_;            /** Total number of elements in the tensor */
    size_t total_bytes_;            /** Total number of bytes required by the tensor */
    Device device_;                 /** The current device where the tensor is stored */
    Device head_;                   /** Tracks the device with the latest valid data */

    T* cpu_data_ptr_;               /** Pointer to the data stored in CPU memory */
    T* gpu_data_ptr_;               /** Pointer to the data stored in GPU memory */

    /**
     * @brief Allocates memory for the tensor in CPU memory.
     */
    void AllocateCPUMemory();

    /**
     * @brief Allocates memory for the tensor in GPU memory.
     */
    void AllocateGPUMemory();

    /**
     * @brief Converts multi-dimensional indices to a flat index for 1D memory access.
     * 
     * @param indices A vector of indices specifying the location in the tensor.
     * @return The flat index corresponding to the given multi-dimensional indices.
     */
    size_t FlattenIndex(const std::vector<size_t>& indices) const;
  };

} // namespace oris_ai

#endif  // INCLUDE_ORIS_AI_COMMON_TENSOR_H_
