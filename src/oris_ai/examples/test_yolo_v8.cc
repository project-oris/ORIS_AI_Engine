/*******************************************************************************
 * Copyright (c) 2024 Electronics and Telecommunications Research Institute (ETRI)
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
#include "oris_ai/model/model.h"

#include <chrono>  // for timing
#include <cuda_runtime.h> // for cudaDeviceSynchronize
#include <algorithm>  // for std::clamp

std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

// Colors array for each object class
std::vector<cv::Scalar> colors = {
  cv::Scalar(0, 255, 0),      // Green
  cv::Scalar(0, 0, 255),      // Red
  cv::Scalar(189, 114, 0),    // Blue
  cv::Scalar(32, 177, 237),   // Yellow
  cv::Scalar(142, 47, 126),   // Purple
  cv::Scalar(25, 83, 217),    // Orange  
  cv::Scalar(238, 190, 77),   // Cyan
  cv::Scalar(153, 153, 153),  // Gray
  cv::Scalar(14, 127, 255),   // Light Orange
  cv::Scalar(34, 189, 188)    // Olive
};

// Get color based on class_id
cv::Scalar getColor(int class_id) {
  int index = class_id % colors.size();
  return colors[index];
}

struct LetterBoxInfo {
  float scale_w = 1.0f;  // width scale ratio
  float scale_h = 1.0f;  // height scale ratio
  int pad_x = 0;         // left padding
  int pad_y = 0;         // top padding
};

// Unified letterbox that supports both fixed resize and auto padding.
void letterbox(const cv::Mat& image, cv::Mat& image_padded, LetterBoxInfo& info,
    bool auto_pad = true, bool scale_fill = false,
    const cv::Size& target_size = cv::Size(640, 640),
    bool scale_up = true, bool center = true, int stride = 32,
    const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
  cv::Size shape = image.size();

  cv::Size new_shape;
  float dw = 0.0f, dh = 0.0f;

  if (scale_fill) {  // stretch to target
    new_shape = target_size;
    info.scale_w = static_cast<float>(new_shape.width) / shape.width;
    info.scale_h = static_cast<float>(new_shape.height) / shape.height;
  } else {
    float r = std::min(static_cast<float>(target_size.width) / shape.width,
                      static_cast<float>(target_size.height) / shape.height);
    if (!scale_up)
      r = std::min(r, 1.0f);
    new_shape = cv::Size(static_cast<int>(std::round(shape.width * r)),
                         static_cast<int>(std::round(shape.height * r)));
    info.scale_w = info.scale_h = r;
    dw = target_size.width - new_shape.width;
    dh = target_size.height - new_shape.height;
    if (auto_pad) {
      dw = std::fmod(dw, static_cast<float>(stride));
      dh = std::fmod(dh, static_cast<float>(stride));
    }
  }

  if (center) {
    dw /= 2.0f;
    dh /= 2.0f;
  }

  cv::Mat image_resized;
  if (shape != new_shape)
    cv::resize(image, image_resized, new_shape, 0, 0, cv::INTER_LINEAR);
  else
    image_resized = image.clone();

  int top = center ? static_cast<int>(std::round(dh - 0.1f)) : 0;
  int bottom = center ? static_cast<int>(std::round(dh + 0.1f)) : 0;
  int left = center ? static_cast<int>(std::round(dw - 0.1f)) : 0;
  int right = center ? static_cast<int>(std::round(dw + 0.1f)) : 0;

  info.pad_x = left;
  info.pad_y = top;

  cv::copyMakeBorder(image_resized, image_padded, top, bottom, left, right,
                    cv::BORDER_CONSTANT, color);
}

// Unified post-processing to map boxes back to the original image and draw them.
void draw_detections(const std::vector<oris_ai::Detection>& results, cv::Mat& output_image,
                    const LetterBoxInfo& info) {
  const int org_w = output_image.cols;
  const int org_h = output_image.rows;
  for (const auto& detection : results) {
    int class_id = detection.class_id;
    float confidence = detection.confidence;

    cv::Scalar color = getColor(class_id);

    int x1 = static_cast<int>(std::clamp((detection.x1 - info.pad_x) / info.scale_w,
                                          0.0f, static_cast<float>(org_w)));
    int y1 = static_cast<int>(std::clamp((detection.y1 - info.pad_y) / info.scale_h,
                                          0.0f, static_cast<float>(org_h)));
    int x2 = static_cast<int>(std::clamp((detection.x2 - info.pad_x) / info.scale_w,
                                          0.0f, static_cast<float>(org_w)));
    int y2 = static_cast<int>(std::clamp((detection.y2 - info.pad_y) / info.scale_h,
                                          0.0f, static_cast<float>(org_h)));

    std::cout << "Class ID: " << class_id << ", Confidence: " << confidence
              << ", Box: (" << x1 << ", " << y1 << "), (" << x2 << ", " << y2
              << ")" << std::endl;

    cv::rectangle(output_image, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

    std::string label = classes[class_id] + ": " +
      std::to_string(confidence * 100).substr(0, 5) + "%";
    int baseline;
    cv::Size textSize =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    y1 = std::max(y1, textSize.height);
    cv::rectangle(output_image,
                  cv::Point(x1, y1 - textSize.height - baseline),
                  cv::Point(x1 + textSize.width, y1), color, cv::FILLED);
    cv::putText(output_image, label, cv::Point(x1, y1 - baseline),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }
}

void PrintUsage(const std::string& program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]\n";
  std::cout << "Options:\n";
  std::cout << "  -c      Use CPU for inference. Default is GPU.\n";
  std::cout << "  -sync   Synchronize CUDA before timing. (For accurate GPU timing)\n";
  std::cout << "Example:\n";
  std::cout << "  " << program_name << " -c\n";
  std::cout << "  " << program_name << " -sync\n";
}

int main(int argc, char* argv[]) {
  PrintUsage(argv[0]);
  // Set default device to GPU
  auto device = oris_ai::Device::GPU;

  bool use_cuda_sync = false;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-c") {
      device = oris_ai::Device::CPU;
    } else if (arg == "-sync") {
      use_cuda_sync = true;
    } else {
      return 1; // Invalid argument
    }
  }

  std::cout << "Inference device: " 
            << (device == oris_ai::Device::CPU ? "CPU" : "GPU - CUDA & cuDNN") << std::endl;
  if (device == oris_ai::Device::GPU && use_cuda_sync) {
    std::cout << "CUDA synchronization enabled for timing." << std::endl;
  }

  oris_ai::ModelType model_type = oris_ai::ModelType::YOLOv8n;
  oris_ai::TaskType task_type = oris_ai::TaskType::Detection;

  // If the size of the image you want to recognize the object on is not the same,
  // use the scale_fill option in letterbox to forcibly resize it to 640 * 640.
  //  - letterbox(converted_img, input_image, false, true)
  constexpr size_t detect_height = 640;
  constexpr size_t detect_width = 640;

  // Create a model for detection
  auto model = oris_ai::CreateModel<float>(model_type, task_type, device);
  model->CreateInputImageTensor(detect_height, detect_width, 255.0f);
  model->Open("../../test_model/yolov8n_oris_bn_merged.pb");

  // GPU warmup with dummy data
  if (device == oris_ai::Device::GPU) {
    std::cout << "Warming up GPU with 5 iterations..." << std::endl;
    cv::Mat dummy_image(detect_height, detect_width, CV_8UC3, cv::Scalar(114, 114, 114));
    for (int w = 0; w < 5; ++w) {
      model->SetInputImageTensor(dummy_image);
      model->Forward();
    }
    cudaDeviceSynchronize();
    std::cout << "Warmup complete." << std::endl;
  }

  cv::Mat org_image;
  cv::Mat converted_img;
  cv::Mat input_image;
  cv::Mat output_image;

  std::chrono::high_resolution_clock::time_point pre_start, pre_stop;
  std::chrono::high_resolution_clock::time_point inference_start, inference_stop;
  std::chrono::high_resolution_clock::time_point post_start, post_stop;
  std::chrono::duration<double, std::milli> elapsed;

  std::string image_path[] = {"../../test_image/bus.jpg", "../../test_image/zidane.jpg"};

  const int num_images = sizeof(image_path) / sizeof(image_path[0]);

  // Cache letterbox metadata since all inputs share dimensions
  LetterBoxInfo lb_info;
  for(int i=0; i<num_images; i++) {
    std::cout << "--------------------------------" << std::endl;
    // Step 1: Load and preprocess image
    org_image = cv::imread(image_path[i]);

    if (org_image.empty()) {
      fprintf(stderr, "Cannot load an image file!\n");
      return -1;
    }

    output_image = org_image.clone();

    cv::cvtColor(org_image, converted_img, cv::COLOR_BGR2RGB);  // Convert BGR to RGB
    letterbox(converted_img, input_image, lb_info, false, true);

    // Step 2: Run inference
    pre_start = std::chrono::high_resolution_clock::now();
    model->SetInputImageTensor(input_image);
    if (device == oris_ai::Device::GPU && use_cuda_sync) {
      cudaDeviceSynchronize();
    }
    pre_stop = std::chrono::high_resolution_clock::now();

    inference_start = std::chrono::high_resolution_clock::now();
    model->Forward();
    if (device == oris_ai::Device::GPU && use_cuda_sync) {
      cudaDeviceSynchronize();
    }
    inference_stop = std::chrono::high_resolution_clock::now();

    post_start = std::chrono::high_resolution_clock::now();
    model->PostProcess();
    post_stop = std::chrono::high_resolution_clock::now();

    // Step 3: Timing results
    elapsed = pre_stop - pre_start;
    std::cout << "Pre-processing time: " << elapsed.count() << " ms" << std::endl;
    elapsed = inference_stop - inference_start;
    std::cout << "Inference time: " << elapsed.count() << " ms" << std::endl;
    elapsed = post_stop - post_start;
    std::cout << "Post-processing time: " << elapsed.count() << " ms" << std::endl;
    const auto& results = model->GetDetectionResults();

    // Step 4: Process results
    draw_detections(results, output_image, lb_info);

    // Save output image with detections drawn
    std::string output_path = "result_" + std::to_string(i) + ".jpg";
    cv::imwrite(output_path, output_image);
    std::cout << "Result saved to " << output_path << std::endl;
  }

  return 0;
}
