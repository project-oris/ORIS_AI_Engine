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

template <typename HandleT>
// Non-owning view of an accelerator handle owned elsewhere.
class BorrowedResource {
 public:
  BorrowedResource() = default;
  explicit BorrowedResource(HandleT h) : handle_(h) {}

  ~BorrowedResource() = default;
  BorrowedResource(const BorrowedResource&)            = default;
  BorrowedResource& operator=(const BorrowedResource&) = default;
  BorrowedResource(BorrowedResource&&)                 = default;
  BorrowedResource& operator=(BorrowedResource&&)      = default;

  HandleT  get() const { return handle_; }
  bool     valid() const { return handle_ != nullptr; }
  explicit operator bool() const { return handle_ != nullptr; }

 private:
  HandleT handle_ = nullptr;
};

}  // namespace oris_ai
