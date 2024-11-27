/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include "oris_ai/layer/maxpooling.h"

namespace oris_ai {

template <typename T>
class MaxPoolingCPU : public MaxPooling<T> {
  public:
    MaxPoolingCPU(const std::string& name) : MaxPooling<T>(name) {}

    ~MaxPoolingCPU() {}

    void Forward() override;
};

} // namespace oris_ai
