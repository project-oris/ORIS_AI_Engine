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

ORIS AI also supports the following models.

| Model family | Task |
|---|---|
| Qwen3.5 2B / 4B / 9B, 35B-A3B | Text and vision-language |
| MiniCPM-V 4.5 / 4.6 | Vision-language |
| Gemma4 E4B | Text and vision-language |
| Llama 3.2 3B | Text generation |
| RF-DETR Nano / Small / Medium | Detection and instance segmentation |
| EfficientViT-B0 | Classification |
| ByteTrack / BoT-SORT with ReID | Multi-object tracking |

For details on using these models, see [Inquiries](#inquiries).

## Model files

The runner reads the Qwen3-VL GGUF files that Qwen publishes:
https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF and
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF. Download the language
model file and the mmproj file of the same model.

| Model | File | Size (bytes) | SHA-256 |
|---|---|---|---|
| 2B Instruct | `Qwen3VL-2B-Instruct-Q8_0.gguf` | 1834427424 | `1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2` |
| 2B Instruct | `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` | 445053216 | `f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82` |
| 8B Instruct | `Qwen3VL-8B-Instruct-Q8_0.gguf` | 8709519456 | `0d264b3941185d00a74f75c4245521dae088ff1efc90ab8d1754e83f5844adb0` |
| 8B Instruct | `mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf` | 752289728 | `c6ba85508d82f42590e6eb77d5340369ab6fecf107a7561d809523d8aa5f3bfd` |

The model files are not part of this repository. Qwen releases them under
the Apache License 2.0; see the model pages above for their terms.

## License

ORIS AI is released under the MIT License. See [LICENSE](LICENSE).
Third-party components and their licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The release in preparation decodes images with stb_image and uses only the
core and imgproc modules of the OpenCV installed on the platform
(Apache-2.0); it does not use the OpenCV image and video codecs. An OpenCV
build can include codecs under other licenses or patent terms, some with
conditions for commercial use; their terms come with the OpenCV build that
you install.

## Inquiries

For details on using ORIS AI, or on features not included in this release,
see https://github.com/project-oris.

## Acknowledgement

This work was supported by Institute of Information & communications
Technology Planning & Evaluation (IITP) grant funded by the Korea
government(MSIT) (No. 2024-00339187, Core Technology Development of
On-device Robot Intelligence SW Platform).
