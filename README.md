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

## Target platform

The release in preparation targets the following platform and uses these
platform libraries at run time.

| Component | Version |
|---|---|
| Device | NVIDIA Jetson AGX Orin 64GB |
| Jetson Linux | R36.4.7 (JetPack 6.2.1, Ubuntu 22.04) |
| CUDA runtime | 12.6 (12.6.68) |
| cuBLAS, cuBLASLt | 12.6.1.4 |
| OpenCV core, imgproc | 4.8.0 |
| glog | 0.4.0 (Ubuntu package 0.5.0+really0.4.0) |

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

## Running Qwen3-VL

The `run_qwen3vl` binary of the release in preparation answers one prompt
about one image:

```sh
run_qwen3vl --llm Qwen3VL-2B-Instruct-Q8_0.gguf \
  --mmproj mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
  --image photo.jpg --prompt "Describe this image in one sentence."
```

| Option | Meaning |
|---|---|
| `--llm <path>` | Language model file (required) |
| `--mmproj <path>` | mmproj file of the same model (required) |
| `--image <path>` | Input image, JPEG or PNG (required) |
| `--prompt <text>` | User prompt (default: "Describe this image in one sentence.") |
| `-n <int>` | Maximum generated tokens (default 32) |
| `--greedy` | Deterministic argmax selection (the default) |
| `--temperature <float>`, `--top-k <int>`, `--top-p <float>` | Sampling; any of them selects sampling unless `--greedy` is given |
| `--seed <int>` | Sampler seed (default 0) |
| `--max-prefill <int>` | Longest accepted prompt in tokens, image included (default 2048) |
| `--max-seq <int>` | KV cache length: prompt and generated tokens (default 4096) |
| `--prefill-chunk <int>` | Largest prompt piece per forward pass (default 1024; 0 = whole prompt) |
| `--kv-initial-tokens <int>` | Initial KV cache rows, grown up to `--max-seq` (default 1024; 0 = `--max-seq` at start) |

### Input limits

| Item | Limit |
|---|---|
| Conversation | One turn with one image and one prompt; the system prompt is "You are a helpful assistant" |
| Image | Decoded with stb_image; EXIF orientation is not applied |
| Image tokens | 8 to 4096 per image; each token covers 32 x 32 pixels |
| Resize | The aspect ratio is kept, both sides are rounded to multiples of 32 and the area is fitted into 8192 to 4194304 pixels |
| `--max-prefill` | Raised to the image tokens + 512 when it is smaller |
| `--max-seq` | Raised to the image tokens + 512 + `-n` when it is smaller; keep it within the model context of 262144 tokens |
| Generation | Stops at the end-of-turn token or after `-n` tokens |

### Output

The runner prints the model, the sampling settings, the image size and its
image tokens, the generated text, why the generation ended (`eog` or
`token-limit`), the number of generated tokens, and timing rows in the
format of the llama.cpp performance report. The total time runs from the
start of the request to the last token and includes an untimed warmup pass.

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 2 | Invalid or missing option |
| 7 | No `--image` |
| 11 | The model files are not a Qwen3-VL Instruct Q8_0 pair |
| 13 | The image cannot be read, or the prompt does not fit the limits |
| 14 | Out of memory, or the run stopped on an internal error |

## Public headers

The headers under [include/](include) describe the ORIS AI runtime that is
built into `run_qwen3vl`, as a reference for its API. The release has no
library to link with them. They are fixed to the build configuration of the
release, so they carry no build options, and their API may change between
releases.

## Runtime notes

- The release runs Qwen3-VL without the device-specific optimizations of
  ORIS AI; for optimized builds, see [Inquiries](#inquiries).
- The runner only reads the model files; it writes nothing next to them.
- `ORIS_*` environment variables set by the user are ignored when
  `run_qwen3vl` runs. The variables named in the headers are for
  development builds.

## Troubleshooting

| Exit code | What to check |
|---|---|
| 11 | The file names and SHA-256 values in [Model files](#model-files), and that `--llm` and `--mmproj` belong to the same model |
| 13 | That the image is a JPEG or PNG file; for a long prompt, a larger `--max-prefill` or `--max-seq` |
| 14 | Smaller `--max-seq`, `--max-prefill` or `--prefill-chunk`, a smaller image, or fewer other GPU programs |

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
