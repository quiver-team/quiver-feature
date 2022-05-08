#pragma once

#include <qvf/common.h>
#include <iostream>

namespace qvf {
class RegisteredMemoryBuffer {
 private:
  uint64_t total_size_in_bytes;
  void* base_address;

 public:
  RegisteredMemoryBuffer();

  void* alloc(uint64_t size_in_bytes) {
    QUIVER_FEATURE_ASSERT(size_in_bytes <= total_size_in_bytes,
                          "Requested size is larger than buffer size");
    return base_address;
  }
};

}  // namespace qvf
