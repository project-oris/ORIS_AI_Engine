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
      ConvolutionCPU() {}

      ~ConvolutionCPU() {}

      void Forward();
  };

} // namespace oris_ai