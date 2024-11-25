#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "oris_ai/common/tensor.h"
#include "oris_ai/protobuf/oris_ai_yolov8.pb.h"

int main() {
  // Open the protobuf file
  std::fstream input("/Data/Dev/ultralytics/oris_ai_convertor/yolov8n_oris_bn_merged.pb", std::ios::in | std::ios::binary);
  if (!input) {
    std::cerr << "Cannot open file: yolov8n_oris_bn_merged.pb" << std::endl;
    return -1;
  }

  // Create Model object and read data from protobuf file
  oris_ai::TorchModel model;
  if (!model.ParseFromIstream(&input)) {
    std::cerr << "Failed to read protobuf file." << std::endl;
    return -1;
  }

  // Find the first Conv2d layer
  const oris_ai::TorchLayer& first_layer = model.layers(0);  // Get the first layer
  if (!first_layer.has_conv2d()) {
    std::cerr << "The first layer is not a Conv2d." << std::endl;
    return -1;
  }

  // Check Conv2d data
  const oris_ai::TorchConv2d& conv2d = first_layer.conv2d();
  std::cout << "in_channels: " << conv2d.in_channels() << std::endl;
  std::cout << "out_channels: " << conv2d.out_channels() << std::endl;
  std::cout << "kernel_size: ";
  for (int size : conv2d.kernel_size()) std::cout << size << " ";
  std::cout << std::endl;

  std::cout << "stride: ";
  for (int size : conv2d.stride()) std::cout << size << " ";
  std::cout << std::endl;

  std::cout << "padding: ";
  for (int size : conv2d.padding()) std::cout << size << " ";
  std::cout << std::endl;

  std::cout << "dilation: ";
  for (int size : conv2d.dilation()) std::cout << size << " ";
  std::cout << std::endl;

  std::cout << "groups: " << conv2d.groups() << std::endl;
  std::cout << "bias: " << (conv2d.use_bias() ? "true" : "false") << std::endl;

  // Print size of weight and bias
  std::cout << "weight size: " << conv2d.weight_size() << std::endl;
  std::cout << "bias size: " << conv2d.bias_size() << std::endl;
  
  // Save weight
  std::vector<size_t> weight_shape = {static_cast<size_t>(conv2d.out_channels()), 
                                      static_cast<size_t>(conv2d.in_channels()), 
                                      static_cast<size_t>(conv2d.kernel_size(0)), 
                                      static_cast<size_t>(conv2d.kernel_size(1))};
  oris_ai::Tensor<float> weight_tensor(weight_shape);

  // Get a pointer to the weight data via GetCPUDataPtr() and 
  // use std::copy to copy the data
  float* weight_data_ptr = weight_tensor.GetCPUDataPtr();
  std::copy(conv2d.weight().begin(), conv2d.weight().end(), weight_data_ptr);

  // Save bias
  std::vector<size_t> bias_shape = {static_cast<size_t>(conv2d.out_channels())};
  oris_ai::Tensor<float> bias_tensor(bias_shape);

  // Get a pointer to the bias data via GetCPUDataPtr() and 
  // use std::copy to copy the data
  float* bias_data_ptr = bias_tensor.GetCPUDataPtr();
  std::copy(conv2d.bias().begin(), conv2d.bias().end(), bias_data_ptr);

  return 0;
}
