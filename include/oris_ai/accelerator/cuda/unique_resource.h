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

namespace oris_ai {

template <typename Traits>
// Move-only owner of an accelerator handle. Traits supply Handle, Create()
// and Destroy().
class UniqueResource {
 public:
  using Handle = typename Traits::Handle;

  UniqueResource() = default;

  static UniqueResource Create() {
    UniqueResource r;
    r.handle_ = Traits::Create();
    return r;
  }

  ~UniqueResource() { reset(); }

  UniqueResource(UniqueResource&& o) noexcept : handle_(o.handle_) {
    o.handle_ = nullptr;
  }

  UniqueResource& operator=(UniqueResource&& o) noexcept {
    if (this != &o) {
      reset();
      handle_ = o.handle_;
      o.handle_ = nullptr;
    }
    return *this;
  }

  UniqueResource(const UniqueResource&)            = delete;
  UniqueResource& operator=(const UniqueResource&) = delete;

  void reset() {
    if (handle_ != nullptr) {
      Traits::Destroy(handle_);
      handle_ = nullptr;
    }
  }

  Handle   get() const { return handle_; }
  bool     valid() const { return handle_ != nullptr; }
  explicit operator bool() const { return handle_ != nullptr; }

 private:
  Handle handle_ = nullptr;
};

}  // namespace oris_ai
