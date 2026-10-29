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
#pragma once

#include "oris_ai/gguf_reader/gguf_types.h"

#include <atomic>
#include <cstdint>
#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>
#include <sys/mman.h>


namespace oris_ai {

class MmapReclaimQueue;
class MmapReclaimEpoch;

// Mappings of weight-layout caches that hold copies of a reader's tensors.
// The owner removes a mapping before it unmaps it, never inside a
// CollectionScope.
class ReclaimCompanions {
 public:
  class CollectionScope {
   public:
    explicit CollectionScope(const ReclaimCompanions& registry) : registry_(registry) {
      registry_.collecting_.fetch_add(1, std::memory_order_acq_rel);
    }
    ~CollectionScope() { registry_.collecting_.fetch_sub(1, std::memory_order_acq_rel); }
    CollectionScope(const CollectionScope&) = delete;
    CollectionScope& operator=(const CollectionScope&) = delete;

   private:
    const ReclaimCompanions& registry_;
  };

  void Add(const void* base, size_t bytes);
  void Remove(const void* base);
  std::vector<std::pair<const void*, size_t>> Snapshot() const;
  CollectionScope BeginCollection() const { return CollectionScope(*this); }
  int collections_in_flight() const { return collecting_.load(std::memory_order_acquire); }

 private:
  mutable std::mutex mutex_;
  std::vector<std::pair<const void*, size_t>> mappings_;
  mutable std::atomic<int> collecting_{0};
};

// GGUF file reader with an mmap of the tensor data section.
class GGUFReader {
public:
  ~GGUFReader();

  bool read_gguf(const std::string& filename);

  // Call after read_gguf().
  bool mmap_data_section();

  // nullptr until the data section is mapped.
  const void* get_tensor_data(const TensorInfo& info) const;

  // The offset is relative to get_data_base(). Works after
  // ReleaseParsingState(). nullptr when unmapped or out of bounds.
  const void* get_data_span(size_t offset, size_t bytes) const;

  // Frees the metadata and the tensor list. The mapping stays for borrowed
  // pointers.
  void ReleaseParsingState();

  // The queue outlives the encoder. Drain it before rereading weights.
  std::shared_ptr<MmapReclaimQueue> GetOrCreateMmapReclaimQueue(size_t byte_limit);
  const std::shared_ptr<MmapReclaimQueue>& mmap_reclaim_queue() const {
    return mmap_reclaim_queue_;
  }
  void WaitForMmapReclaims();
  // Drain the submitted jobs before a new epoch.
  std::shared_ptr<MmapReclaimEpoch> GetOrCreateMmapReclaimEpoch();
  void BeginMmapReclaimEpoch();
  // Byte counts and admission wait of the source page reclaim.
  struct SourceReclaimStats {
    std::atomic<size_t> submitted_bytes{0};
    std::atomic<size_t> queued_bytes{0};
    std::atomic<size_t> succeeded_bytes{0};
    std::atomic<size_t> failed_bytes{0};
    std::atomic<uint64_t> admission_wait_ns{0};
  };
  const std::shared_ptr<SourceReclaimStats>& source_reclaim_stats() const {
    return source_reclaim_stats_;
  }

  // mmap_fd() is -1 until mmap_data_section() has run.
  const std::string& filename() const { return filename_; }
  int mmap_fd() const { return mmap_fd_; }

  // Copies of this file's tensors, reclaimed with its mapping. The registry is
  // shared, so an owner may outlive this reader.
  const std::shared_ptr<ReclaimCompanions>& reclaim_companion_registry() const {
    return reclaim_companions_;
  }
  std::vector<std::pair<const void*, size_t>> reclaim_companions() const;

  void* get_mmap_data() const { return mmap_data_; }
  size_t get_mmap_size() const { return mmap_size_; }
  // Start of the tensor data section in the mapping. TensorInfo offsets are
  // relative to it.
  size_t get_data_offset() const { return data_offset_; }
  const uint8_t* get_data_base() const {
    return static_cast<const uint8_t*>(mmap_data_) + data_offset_;
  }

  const std::vector<TensorInfo>& get_tensors() const { return tensors_; }
  size_t get_tensor_count() const { return tensors_.size(); }

  bool get_metadata_uint32(const std::string& key, uint32_t& value) const {
    auto it = metadata_uint32_.find(key);
    if (it != metadata_uint32_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_int32(const std::string& key, int32_t& value) const {
    auto it = metadata_int32_.find(key);
    if (it != metadata_int32_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_uint64(const std::string& key, uint64_t& value) const {
    auto it = metadata_uint64_.find(key);
    if (it != metadata_uint64_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_float32(const std::string& key, float& value) const {
    auto it = metadata_float32_.find(key);
    if (it != metadata_float32_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_string(const std::string& key, std::string& value) const {
    auto it = metadata_string_.find(key);
    if (it != metadata_string_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_string_array(const std::string& key, std::vector<std::string>& value) const {
    auto it = metadata_string_array_.find(key);
    if (it != metadata_string_array_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_bool(const std::string& key, bool& value) const {
    auto it = metadata_bool_.find(key);
    if (it != metadata_bool_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_uint32_array(const std::string& key, std::vector<uint32_t>& value) const {
    auto it = metadata_uint32_array_.find(key);
    if (it != metadata_uint32_array_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_int32_array(const std::string& key, std::vector<int32_t>& value) const {
    auto it = metadata_int32_array_.find(key);
    if (it != metadata_int32_array_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_bool_array(const std::string& key, std::vector<bool>& value) const {
    auto it = metadata_bool_array_.find(key);
    if (it != metadata_bool_array_.end()) { value = it->second; return true; }
    return false;
  }
  bool get_metadata_float32_array(const std::string& key,
                                  std::vector<float>& value) const {
    auto it = metadata_float32_array_.find(key);
    if (it != metadata_float32_array_.end()) { value = it->second; return true; }
    return false;
  }

protected:
  std::ifstream file_;
  uint64_t file_size_ = 0;
  uint32_t version_;
  int64_t n_tensors_;
  int64_t n_kv_;
  size_t alignment_;
  size_t data_offset_;
  std::vector<TensorInfo> tensors_;

  std::map<std::string, uint32_t> metadata_uint32_;
  std::map<std::string, int32_t>  metadata_int32_;
  std::map<std::string, uint64_t> metadata_uint64_;
  std::map<std::string, float> metadata_float32_;
  std::map<std::string, std::string> metadata_string_;
  std::map<std::string, std::vector<std::string>> metadata_string_array_;
  std::map<std::string, bool> metadata_bool_;
  std::map<std::string, std::vector<uint32_t>> metadata_uint32_array_;
  std::map<std::string, std::vector<int32_t>> metadata_int32_array_;
  std::map<std::string, std::vector<bool>> metadata_bool_array_;
  std::map<std::string, std::vector<float>> metadata_float32_array_;
  std::map<std::string, gguf_type> metadata_types_;

  void* mmap_data_ = nullptr;
  size_t mmap_size_ = 0;
  int mmap_fd_ = -1;
  std::string filename_;

private:
  std::shared_ptr<ReclaimCompanions> reclaim_companions_ =
      std::make_shared<ReclaimCompanions>();
  std::shared_ptr<MmapReclaimQueue> mmap_reclaim_queue_;
  std::shared_ptr<MmapReclaimEpoch> mmap_reclaim_epoch_;
  std::shared_ptr<SourceReclaimStats> source_reclaim_stats_ =
      std::make_shared<SourceReclaimStats>();
  bool mmap_reclaim_epoch_unavailable_ = false;
  bool mmap_reclaim_queue_unavailable_ = false;
  bool ValidateRopeScalingMetadata(const std::string& architecture,
                                   std::string* error = nullptr) const;

  template<typename T>
  bool read(T& value);
  bool read_string(std::string& str);
  bool skip_kv_pairs();
  bool skip_value(gguf_type type);
  bool check_array_len(uint64_t arr_len);
  bool tensor_range_in_mmap(const TensorInfo& info) const;
};

}  // namespace oris_ai
