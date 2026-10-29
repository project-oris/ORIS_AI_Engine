/*******************************************************************************
 * Copyright (c) 2025 Electronics and Telecommunications Research Institute (ETRI)
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

#include <glog/logging.h>

#include <memory>
#include <string>
#include <utility>

#include "oris_ai/gguf_reader/gguf_reader.h"

namespace oris_ai {

// GGUF open steps. Each logs its failure under the caller's tag and returns
// false.

// On failure *reader_out holds the reader that failed.
inline bool ReadGgufHeader(const std::string& path, const char* tag,
                           std::unique_ptr<GGUFReader>* reader_out) {
  *reader_out = std::make_unique<GGUFReader>();
  if (!(*reader_out)->read_gguf(path)) {
    LOG(ERROR) << tag << "read_gguf failed for " << path;
    return false;
  }
  return true;
}

inline bool MapGgufDataSection(GGUFReader* reader, const std::string& path,
                               const char* tag) {
  if (!reader->mmap_data_section()) {
    LOG(ERROR) << tag << "mmap_data_section failed for " << path;
    return false;
  }
  return true;
}

// *parser_out holds the parser even when the parse fails.
template <typename ParserT>
bool ParseGgufHeaderWith(GGUFReader* reader, bool (ParserT::*parse)(),
                         const char* tag, const char* parser_type_name,
                         const char* parse_name,
                         std::unique_ptr<ParserT>* parser_out) {
  *parser_out = std::make_unique<ParserT>(reader);
  if (!((*parser_out).get()->*parse)()) {
    LOG(ERROR) << tag << parser_type_name << "::" << parse_name << " failed";
    return false;
  }
  return true;
}

// Maps the data section of a pair parsed elsewhere. The parser must reference
// the reader. On failure nothing is stored.
template <typename ParserT>
bool AdoptParsedGguf(std::unique_ptr<GGUFReader> reader,
                     std::unique_ptr<ParserT> parser, const std::string& path,
                     const char* tag, std::unique_ptr<GGUFReader>* reader_out,
                     std::unique_ptr<ParserT>* parser_out) {
  if (reader == nullptr || parser == nullptr ||
      parser->reader() != reader.get()) {
    LOG(ERROR) << tag << "prepared parser does not reference its reader ("
               << path << ")";
    return false;
  }
  if (!MapGgufDataSection(reader.get(), path, tag)) return false;
  *reader_out = std::move(reader);
  *parser_out = std::move(parser);
  return true;
}

}  // namespace oris_ai
