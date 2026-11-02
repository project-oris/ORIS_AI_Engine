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

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace oris_ai {
namespace preprocess_util {

// Padding of the resize target.
enum class PillowResizePadding {
  kNone,
  kCeil,
};

// Bicubic resizer for three-channel uint8 images; one caller at a time.
// Reference: Pillow bicubic resampling.
class PillowBicubicResizer {
 public:
  // Valid until the next Resize; an identity resize returns a view of `src`.
  cv::Mat Resize(const cv::Mat& src, const cv::Size& target,
                 PillowResizePadding padding,
                 const std::array<uint8_t, 3>& pad_bytes = {0, 0, 0});

  size_t plan_build_count() const { return plan_build_count_; }
  size_t cached_plan_count() const { return plans_.size(); }

 private:
  struct AxisPlan {
    int input_size = 0;
    int output_size = 0;
    int kernel_size = 0;
    std::vector<int> bounds;
    std::vector<int32_t> weights;
  };

  struct ResizePlan {
    int source_width = 0;
    int source_height = 0;
    int target_width = 0;
    int target_height = 0;
    AxisPlan horizontal;
    AxisPlan vertical;
    uint64_t last_use = 0;
  };

  static AxisPlan BuildAxisPlan(int input_size, int output_size);
  ResizePlan& FindOrBuildPlan(int source_width, int source_height, int target_width,
                              int target_height);

  std::vector<ResizePlan> plans_;
  std::vector<uint8_t> horizontal_buffer_;
  std::vector<uint8_t> output_buffer_;
  uint64_t use_counter_ = 0;
  size_t plan_build_count_ = 0;
};

}  // namespace preprocess_util
}  // namespace oris_ai
