/****************************************************************************** 
// Copyright 2024 Electronics and Telecommunications Research Institute (ETRI).
// All Rights Reserved.
******************************************************************************/
#include "oris_ai/oss/oss_helper.h"

void PrintUsage(const std::string& program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]\n";
  std::cout << "Options:\n";
  std::cout << "  -c    Use CPU for inference. Default is GPU.\n";
  std::cout << "Example:\n";
  std::cout << "  " << program_name << " -c\n";
}

int main(int argc, char* argv[]) {
  PrintUsage(argv[0]);

  bool use_gpu = true;
  if (argc > 1) {
    std::string arg = argv[1];
    if (arg == "-c") {
      use_gpu = false;
    } else {
      return 1;
    }
  }
  
  Yolov8Detection(use_gpu);
}