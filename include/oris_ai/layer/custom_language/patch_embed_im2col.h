/*******************************************************************************
 * Copyright (c) 2026 Electronics and Telecommunications Research Institute (ETRI)
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

#include "oris_ai/accelerator/cuda/cuda_functions.h"
#include "oris_ai/gguf_reader/gguf_types.h"
#include "oris_ai/tensor/tensor.h"

#include "oris_ai/accelerator/cuda/cublas_manager.h"
#include <cublasLt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace oris_ai {

// Patch embedding by im2col gather and fixed-M FP32 GEMM; GPU only.
template <typename T>
class PatchEmbedIm2Col {
 public:
  explicit PatchEmbedIm2Col(const std::string& layer_name);
  ~PatchEmbedIm2Col();

  // weight0 and weight1 are [embed_dim, C, P, P]; an F32 weight stays borrowed
  // until its first upload. weight1 is optional and added by a second GEMM.
  void Init(size_t in_channels, size_t embed_dim, size_t patch_size,
            size_t max_n_pos,
            const void* weight0, size_t bytes0, ggml_type dtype0,
            const void* weight1 = nullptr, size_t bytes1 = 0,
            ggml_type dtype1 = GGML_TYPE_F32);

  // image_chw is [1, C, H, W]; out receives [n_pos, embed_dim]; the caller sets
  // the view. Pixels outside the patch grid are ignored.
  void Run(Tensor<T>* image_chw, size_t pos_h, size_t pos_w,
           TokenIndexMode mode, Tensor<T>* out);

  Tensor<T>* GetXcolTensor() { return xcol_.get(); }

  // The previous gather scratch must no longer be in use.
  void ResizeEnvelope(size_t max_n_pos);

  // Safe before or after the first Forward.
  void ReleaseHostWeights();

  void InvalidatePlannedStorage();

  // Source bytes stay live through the next weight upload.
  void RebindRawWeights(const void* weight0, size_t bytes0, ggml_type dtype0,
                         const void* weight1 = nullptr, size_t bytes1 = 0,
                         ggml_type dtype1 = GGML_TYPE_F32);

 private:
  template <typename U> friend struct PatchEmbedIm2ColTestAccess;
  std::string layer_name_;

  size_t in_channels_ = 0;
  size_t embed_dim_   = 0;
  size_t patch_size_  = 0;
  size_t max_n_pos_   = 0;
  size_t k_dim_       = 0;

  std::vector<float> weight0_fp32_;
  std::unique_ptr<Tensor<T>> weight0_;
  std::vector<float> weight1_fp32_;
  std::unique_ptr<Tensor<T>> weight1_;

  std::unique_ptr<Tensor<T>> xcol_;

  cublasLtMatmulDesc_t   op_desc_ = nullptr;
  cublasLtMatrixLayout_t a_desc_  = nullptr;
  cublasLtMatrixLayout_t b_desc_  = nullptr;
  cublasLtMatrixLayout_t c_desc_  = nullptr;
  cublasLtMatmulAlgo_t   algo_{};
  void*  work_space_      = nullptr;
  size_t size_work_space_ = 0;

  void CUBLASLtSetup();

  bool is_initialized_ = false;
};

}  // namespace oris_ai
