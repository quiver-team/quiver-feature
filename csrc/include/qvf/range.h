#pragma once

#include <iostream>
namespace qvf {
class Range {
 private:
  uint64_t start;
  uint64_t end;

 public:
  Range() {}
  Range(uint64_t start, uint64_t end) : start(start), end(end) {}
  void set_params(uint64_t start, uint64_t end) {
    this->start = start;
    this->end = end;
  }
  Range& operator=(const Range& other) {
    this->start = other.start;
    this->end = other.end;
    return *this;
  }
  uint64_t range_start() { return start; }
  uint64_t range_end() { return end; }
};
}  // namespace qvf
