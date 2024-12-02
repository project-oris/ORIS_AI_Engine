/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

#include <opencv2/opencv.hpp>

struct Detection {
  int class_id{0};
  float confidence{0.0f};
  cv::Rect box{};
};