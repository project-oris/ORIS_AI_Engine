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

#include "oris_ai/model/vision/pillow_bicubic_resize.h"
#include "oris_ai/tensor/tensor.h"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace oris_ai {

class GGUFReader;

// Target canvas: keep the aspect ratio or force a square.
enum class AspectRatioMode {
  PRESERVE     = 0,
  SQUARE_FORCE = 1,
};

// Image preprocessor: RGB uint8 image to normalized [3, H, W] floats.
class Qwen3ImagePreprocessor {
 public:
  explicit Qwen3ImagePreprocessor(
      AspectRatioMode mode = AspectRatioMode::PRESERVE);
  ~Qwen3ImagePreprocessor() = default;

  // Absent keys keep the Qwen3-VL 2B defaults.
  bool LoadConfig(const GGUFReader& reader);

  // The maximum cannot exceed the model's loaded maximum.
  bool SetTokenLimits(size_t min_tokens, size_t max_tokens);

  // Decodes with stb_image; empty on failure. EXIF orientation is not applied.
  static cv::Mat DecodeImageRGB8(const std::string& path);

  // Empty shape for unsupported input.
  Tensor<float> PreprocessRGB8(const cv::Mat& rgb_image);

  // False, with the outputs unchanged, for a size that has none.
  bool PlanTargetSize(int image_width, int image_height, int* target_width,
                      int* target_height) const;

  size_t patch_size()    const { return patch_size_; }
  size_t spatial_merge() const { return spatial_merge_; }
  size_t max_tokens()    const { return max_tokens_; }

 private:
  template <typename U> friend struct Qwen3ImagePreprocessorTestAccess;

  AspectRatioMode mode_;

  size_t image_size_    = 768;
  size_t patch_size_    = 16;
  size_t spatial_merge_ = 2;

  int image_min_tokens_ = 8;
  int image_max_tokens_ = 4096;
  int image_min_pixels_ = -1;
  int image_max_pixels_ = -1;

  size_t max_tokens_ = 4096;
  size_t model_max_tokens_ = 4096;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> std_  = {0.5f, 0.5f, 0.5f};

  cv::Mat ResizeRGB8(const cv::Mat& image,
                     int& target_w_out, int& target_h_out);

  void FillTensor(const cv::Mat& rgb_u8, int target_h, int target_w,
                  float* dst_ptr);

  preprocess_util::PillowBicubicResizer bicubic_resizer_;
  std::array<std::array<float, 256>, 3> normalization_table_;
};

}  // namespace oris_ai
