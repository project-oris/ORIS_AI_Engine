/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#pragma once

struct Detection {
  int class_id{0};
  float confidence{0.0f};
  float x1{0.0f};
  float y1{0.0f};
  float x2{0.0f};
  float y2{0.0f};
};