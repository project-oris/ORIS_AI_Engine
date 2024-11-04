/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#ifndef INCLUDE_ORIS_AI_COMMON_CUDA_STREAM_MANAGER_H_
#define INCLUDE_ORIS_AI_COMMON_CUDA_STREAM_MANAGER_H_

#include "oris_ai/common/cuda_define.h"

namespace oris_ai {

class CUDAStreamManager {
public:
  /**
   * @brief Default constructor for CUDAStreamManager
   * Initializes the CUDA stream
   */
  CUDAStreamManager();
  
  /**
   * @brief Destructor for CUDAStreamManager
   * Cleans up the CUDA stream
   */
  ~CUDAStreamManager();

  /**
   * @brief Singleton instance getter for CUDAStreamManager
   * Ensures only one instance of the stream manager exists
   * @return Reference to the singleton instance
   */
  static CUDAStreamManager& GetInstance() {
    static CUDAStreamManager instance;
    return instance;
  }

  /**
   * @brief Getter for the CUDA stream
   * @return cudaStream_t stream
   */
  inline cudaStream_t GetStream() const { return stream_; }

private:
  cudaStream_t stream_;  /** CUDA stream */
};

}  // namespace oris_ai

#endif  // INCLUDE_ORIS_AI_COMMON_CUDA_STREAM_MANAGER_H_
