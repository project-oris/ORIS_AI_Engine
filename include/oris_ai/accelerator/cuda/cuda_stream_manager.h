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

#include "oris_ai/accelerator/cuda/cuda_define.h"

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
     * @brief Deleted copy constructor to prevent copying of the singleton instance.
     */
    CUDAStreamManager(const CUDAStreamManager&) = delete;

    /**
     * @brief Deleted copy assignment operator to prevent assignment of the singleton instance.
     */
    CUDAStreamManager& operator=(const CUDAStreamManager&) = delete;

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
