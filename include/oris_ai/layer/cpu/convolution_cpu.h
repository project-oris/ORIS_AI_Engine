/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/convolution.h"

namespace oris_ai {

  template <typename T>
  class ConvolutionCPU : public Convolution<T> {
    public:
      ConvolutionCPU(const std::string& name) : Convolution<T>(name) {}
      ~ConvolutionCPU() {}

      void Forward() override;

    private:
      /**
       * @brief Perform the im2col operation for convolution.
       */
      void Im2Col(const T* input_data, T* col_data, int input_h, int input_w,
                  int kernel_h, int kernel_w, int pad_h, int pad_w,
                  int stride_h, int stride_w);
  };

} // namespace oris_ai