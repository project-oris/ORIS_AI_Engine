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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct llama_model;
struct llama_vocab;
using llama_token = int32_t;

namespace oris_ai {

class GGUFReader;

// kParse: a special-token spelling emits its id (trusted text).
// kContain: spellings are ordinary bytes (user text).
enum class SpecialPolicy { kParse, kContain };

// The output ids are best-effort on every status.
enum class TokenizeStatus { kOk, kEmptyInput, kUnrepresentableBytes };

// BPE tokenizer read from GGUF metadata.
// Reference: llama.cpp llama-bpe, qwen2; GPT-2 bytes_to_unicode.
class Tokenizer {
public:
  explicit Tokenizer(const std::string &tokenizer_name);

  ~Tokenizer() = default;

  bool InitFromGGUF(const std::string& model_path);

  // Copies the tables, so `reader` may release its parsing state afterwards.
  bool InitFromMetadata(const GGUFReader& reader);

  // Embedded NUL bytes are data. `out` is cleared and filled on every status.
  TokenizeStatus Tokenize(std::string_view text, SpecialPolicy policy,
                          bool add_bos, std::vector<llama_token>* out);

  // Test-only.
  std::vector<std::string> ApplyBPEForTest(const std::vector<std::string>& symbols) {
    return ApplyBPE(symbols);
  }
  const std::unordered_map<std::string, int32_t>& merges_for_test() const {
    return merges_;
  }

  // SpecialPolicy::kParse, truncated to max_tokens; returns the token count.
  int32_t Tokenize(const char* text, llama_token* tokens, int32_t max_tokens, bool add_bos = true);

  std::string Detokenize(const std::vector<llama_token> &tokens, bool render_special_tokens = true);

  // Empty for a special, unused or invalid token.
  std::string TokenPiece(llama_token token) const;

  // A null `out` is ignored.
  void AppendTokenPiece(llama_token token, std::string* out) const;

  bool IsSpecialToken(llama_token token) const {
    return special_token_ids_.count(token) != 0;
  }

  // GGUF token type 5; renders as nothing.
  bool IsUnusedToken(llama_token token) const {
    return token >= 0 && static_cast<size_t>(token) < token_types_.size() &&
           token_types_[token] == 5;
  }

  llama_token GetBosToken() const { return bos_token_id_; }

  llama_token GetEosToken() const { return eos_token_id_; }

  std::string ApplyChatTemplate(const std::string& user_message, bool conversation_mode = true);

  int GetVocabSize() const { return static_cast<int>(vocab_.size()); }

  // -1 when the spelling is absent.
  llama_token FindTokenId(const std::string& token) const;

private:
  std::string tokenizer_name_;

  std::unordered_map<std::string, llama_token> vocab_;

  std::vector<std::string> id_to_token_;

  // 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte.
  std::vector<int32_t> token_types_;

  std::string tokenizer_pre_;

  std::unordered_map<std::string, int32_t> merges_;

  llama_token bos_token_id_ = -1;
  llama_token eos_token_id_ = -1;

  std::string chat_template_;
  std::unordered_map<std::string, llama_token> special_tokens_;
  std::unordered_set<llama_token> special_token_ids_;

  void AppendTokenPieceUnchecked(llama_token token_id, std::string* out) const;

  std::vector<std::string> ByteEncode(const std::string& text);
  std::vector<std::string> ApplyBPE(const std::vector<std::string>& tokens);

  std::vector<std::string> PreTokenize(const std::string& text) const;

  // 256 and 512 entries; -1 in the second when unmapped.
  static const uint32_t* ByteToUnicodeTable();
  static const int16_t* UnicodeToByteTable();
  static void AppendUTF8(uint32_t cp, std::string* out);
  static bool NextUTF8(const std::string& s, size_t* pos, uint32_t* cp);
};

} // namespace oris_ai
