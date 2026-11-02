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

#include <glog/logging.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>

#include "oris_ai/layer/linear.h"

namespace oris_ai {

// Process-wide count of projections switched to W8A16.
inline std::atomic<int>& VisionQ8W8A16Layers() {
  static std::atomic<int> layers{0};
  return layers;
}

// Projections of the latest tower on W8A16; -1 when none recorded.
inline std::atomic<int>& VisionQ8W8A16TowerLayers() {
  static std::atomic<int> layers{-1};
  return layers;
}

inline void RecordVisionQ8W8A16Tower(int layers) {
  VisionQ8W8A16Layers() += layers;
  VisionQ8W8A16TowerLayers().store(layers);
}

// Whether the Q8_0 ViT projections run W8A16 instead of MMQ.
inline PlanVisionPrecision ResolveVisionPrecision(PlanVisionPrecision plan_vision) {
  bool on = false;
  int source = 0;
  if (const char* env = std::getenv("ORIS_VIT_Q8_W8A16")) {
    on = env[0] == '1' && env[1] == '\0';
    source = 1;
  } else if (plan_vision != PlanVisionPrecision::kNoPlan) {
    on = plan_vision == PlanVisionPrecision::kW8A16;
    source = 2;
  }
  static std::atomic<int> logged{-1};
  const int state = source * 2 + (on ? 1 : 0);
  if (logged.exchange(state) != state) {
    static const char* const kSources[] = {"default", "env", "plan"};
    LOG(INFO) << "VIT_Q8_W8A16 enabled=" << (on ? 1 : 0)
              << " source=" << kSources[source];
  }
  return on ? PlanVisionPrecision::kW8A16 : PlanVisionPrecision::kMmq;
}

inline size_t VisionQ8FragMaxRowsFromEnv(bool* from_env = nullptr) {
  const char* env = std::getenv("ORIS_VIT_Q8_FRAG");
  if (from_env != nullptr) *from_env = env != nullptr;
  return env != nullptr && env[0] == '0' && env[1] == '\0' ? 0 : kVisionQ8FragMaxRows;
}

inline size_t ResolveVisionQ8FragMaxRows() {
  bool from_env = false;
  const size_t rows = VisionQ8FragMaxRowsFromEnv(&from_env);
  static std::atomic<int> logged{-1};
  const int state = (from_env ? 2 : 0) + (rows != 0 ? 1 : 0);
  if (logged.exchange(state) != state) {
    LOG(INFO) << "VIT_Q8_FRAG enabled=" << (rows != 0 ? 1 : 0) << " max_rows=" << rows
              << " source=" << (from_env ? "env" : "default");
  }
  return rows;
}

inline size_t VisionQ8FragMaxRows(size_t fixed = kVisionQ8FragUnresolved) {
  return fixed != kVisionQ8FragUnresolved ? fixed : ResolveVisionQ8FragMaxRows();
}

inline bool VisionQ8W8A16Requested(PlanVisionPrecision fixed =
                                       PlanVisionPrecision::kNoPlan) {
  if (fixed != PlanVisionPrecision::kNoPlan) return fixed == PlanVisionPrecision::kW8A16;
  return ResolveVisionPrecision(PlanVisionPrecision::kNoPlan) == PlanVisionPrecision::kW8A16;
}

}  // namespace oris_ai
