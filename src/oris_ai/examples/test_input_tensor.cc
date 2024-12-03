#include <opencv2/opencv.hpp>

#include "oris_ai/tensor/tensor.h"

#include <chrono>   // for debug
#include <fstream>  // for debug

// Function to compute image resize scale
float generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
  int origin_w = image.cols;
  int origin_h = image.rows;

  int target_h = target_size[0];
  int target_w = target_size[1];

  float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
  float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
  float resize_scale = std::min(ratio_h, ratio_w);
  return resize_scale;
}

// Function to apply letterbox padding
float letterbox(cv::Mat& input_image, cv::Mat& output_image, const std::vector<int>& target_size) {
  if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
    if (input_image.data == output_image.data) {
      return 1.;
    } else {
      output_image = input_image.clone();
      return 1.;
    }
  }

  float resize_scale = generate_scale(input_image, target_size);
  int new_shape_w = std::round(input_image.cols * resize_scale);
  int new_shape_h = std::round(input_image.rows * resize_scale);
  float padw = (target_size[1] - new_shape_w) / 2.;
  float padh = (target_size[0] - new_shape_h) / 2.;

  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);

  cv::resize(input_image, output_image, cv::Size(new_shape_w, new_shape_h),
            0, 0, cv::INTER_AREA);

  cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                      cv::BORDER_CONSTANT, cv::Scalar(114.));
  return resize_scale;
}

// sthong note
// - ConvertMatToTensor()은 사용자 정으에 따라 변경 가능
// - Method 1은 OpenCV를 활용하여 전처리하며, Method 2는 자체 Tensor 구조를 이용하여 정규화함
// - 속도는 큰 차이 없으나, 아주 미세하게 Method 1이 빠름 (소수점 3자리 이하)
// - 비교 대상(ex> LibTorch)에서 사용하는 전처리 알고리즘에 맞게 선택적으로 사용

// // Method 1: 정규화 시 opencv의 내부 함수를 이용
// // 수행시간: 0.06567 ms
// oris_ai::Tensor<float> ConvertMatToTensor(const cv::Mat& input_image, bool rb_swap=false) {
//   // Step 1: Perform BGR -> RGB conversion if necessary
//   cv::Mat converted_image;
//   if (rb_swap)
//     cv::cvtColor(input_image, converted_image, cv::COLOR_BGR2RGB);
//   else
//     converted_image = input_image; // Use input_image as is

//   // Step 2: Create a tensor (in NCHW format)
//   std::vector<size_t> shape = {1, 3, static_cast<size_t>(converted_image.rows), static_cast<size_t>(converted_image.cols)};
//   oris_ai::Tensor<float> input_tensor(shape);

//   float* input_data = input_tensor.GetCPUDataPtr();

//   // Step 3: Wrap the input data
//   std::vector<cv::Mat> input_channels;
//   input_channels.reserve(3);

//   for (unsigned int j = 0; j < 3; ++j) {
//     cv::Mat channel(static_cast<size_t>(converted_image.rows), static_cast<size_t>(converted_image.cols), CV_32FC1, input_data);
//     input_channels.push_back(channel);
//     input_data += static_cast<size_t>(converted_image.rows) * static_cast<size_t>(converted_image.cols);
//   }

//   // Step 4: Normalize and split channels
//   cv::Mat float_image;
//   converted_image.convertTo(float_image, CV_32FC3, 1.0f/255.0f); // Directly convert to float
//   cv::split(float_image, input_channels); // Split channels from the normalized image

//   return input_tensor;
// }

// Method 2: 자체 텐서 생성 후 텐서를 정규화
// 수행시간: 0.068364 ms
oris_ai::Tensor<float> ConvertMatToTensor(const cv::Mat& input_image, bool rb_swap=false) {
  // Step 1: Perform BGR -> RGB conversion if necessary
  cv::Mat converted_image;
  if (rb_swap)
    cv::cvtColor(input_image, converted_image, cv::COLOR_BGR2RGB);
  else
    converted_image = input_image; // Use input_image as is

  // Step 2: Create a tensor (in NCHW format)
  const std::vector<size_t> shape = {1, 3, static_cast<size_t>(converted_image.rows), static_cast<size_t>(converted_image.cols)};

  // Step 3: Create a float tensor directly, converting and normalizing data
  oris_ai::Tensor<float> input_tensor(shape);
  float* tensor_data = input_tensor.GetCPUDataPtr();
  
  // Get the data pointer of OpenCV Mat
  const uint8_t* input_data = converted_image.ptr<uint8_t>(); // Get the pointer to the first row

  // Step 4: Data conversion and normalization (convert NHWC to NCHW)
  const size_t height = shape[2];  // H
  const size_t width = shape[3];   // W
  const size_t channels = shape[1]; // C (RGB)

  // Data conversion and normalization
  const size_t hw_size = height * width;  // Pre-calculate H * W
  const size_t total_pixels = hw_size;    // Total number of pixels (H * W)

  for (size_t c = 0; c < channels; ++c) {
    size_t channel_offset = c * total_pixels; // Offset per channel in NCHW

    for (size_t i = 0; i < total_pixels; ++i) {
      size_t h = i / width; // Current height index
      size_t w = i % width; // Current width index

      size_t nhwc_index = h * width * channels + w * channels + c; // Calculate NHWC index
      tensor_data[channel_offset + i] = static_cast<float>(input_data[nhwc_index]) / 255.0f;
    }
  }

  return input_tensor;
}

int main() {
  // Load example image
  std::string image_path = "../../test_image/bus.jpg";
  cv::Mat image = cv::imread(image_path);

  if (image.empty()) {
    fprintf(stderr, "Cannot load an image file!\n");
    return -1;
  }

  cv::Mat converted_img;
  cv::cvtColor(image, converted_img, cv::COLOR_BGR2RGB);  // Convert BGR to RGB

  cv::Mat input_image;
  letterbox(converted_img, input_image, {640, 640});

  std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

  // Convert image data to Tensor object
  oris_ai::Tensor<float> input_tensor = ConvertMatToTensor(input_image);
  // input_tensor.Permute({0, 3, 1, 2}); // Permute operation is already performed in ConvertMatToTensor

  std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
  std::chrono::duration<double> micro_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Preprocessing Time : " << micro_time.count() << " ms" << std::endl;

  std::cout << "Tensor shape: ";
  const std::vector<size_t>& tensor_shape = input_tensor.GetShape();
  for (size_t dim : tensor_shape) {
    std::cout << dim << " ";
  }
  std::cout << std::endl;

  // Save tensor data to a txt file (for debugging purposes)
  float* tensor_data = input_tensor.GetCPUDataPtr();
  std::ofstream outfile("oris_input_tensor.txt");
  if (outfile.is_open()) {
    const std::vector<size_t>& shape = input_tensor.GetShape();  // Retrieve tensor shape
    size_t total_elements = shape[0] * shape[1] * shape[2] * shape[3];  // N * C * H * W

    for (size_t i = 0; i < total_elements; ++i) {
      outfile << tensor_data[i] << "\n";
    }

    outfile.close();
  } else {
    fprintf(stderr, "Unable to open file for writing\n");
    return -1;
  }

  return 0;
}
