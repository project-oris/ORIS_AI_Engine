# ORIS AI

ORIS AI is a C++17 on-device deep-learning inference framework for vision,
language and vision-language models, developed by the Electronics and
Telecommunications Research Institute (ETRI) in the ORIS (On-device Robot
Intelligence SW-System) project. It runs on NVIDIA GPUs with CUDA.

## Status

A binary release for NVIDIA Jetson AGX Orin 64GB is in preparation. It will
provide a Qwen3-VL command-line runner and the public headers of ORIS AI.

## Models

The release in preparation provides Qwen3-VL 2B and 8B Instruct in Q8_0 for
Jetson AGX Orin 64GB.

ORIS AI also supports the following models through technology transfer.

| Model family | Task |
|---|---|
| Qwen3.5 2B / 4B / 9B, 35B-A3B | Text and vision-language |
| MiniCPM-V 4.5 / 4.6 | Vision-language |
| Gemma4 E4B | Text and vision-language |
| Llama 3.2 3B | Text generation |
| RF-DETR Nano / Small / Medium | Detection and instance segmentation |
| EfficientViT-B0 | Classification |
| ByteTrack / BoT-SORT with ReID | Multi-object tracking |

For technology transfer, see [Inquiries](#inquiries).

## License

ORIS AI is released under the MIT License. See [LICENSE](LICENSE).

## Inquiries

For technology transfer, or for features not included in this release, see
https://github.com/project-oris.

## Acknowledgement

This work was supported by Institute of Information & communications
Technology Planning & Evaluation (IITP) grant funded by the Korea
government(MSIT) (No. 2024-00339187, Core Technology Development of
On-device Robot Intelligence SW Platform).
