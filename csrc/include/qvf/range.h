#pragma once

#include <iostream>
namespace qvf {
class Range {
 private:
  int64_t start;
  int64_t end;

 public:
  Range() {}
  Range(int64_t start, int64_t end) : start(start), end(end) {}
  void set_params(int64_t start, int64_t end) {
    this->start = start;
    this->end = end;
  }
  Range& operator=(const Range& other) {
    this->start = other.start;
    this->end = other.end;
    return *this;
  }
  int64_t range_start() { return start; }
  int64_t range_end() { return end; }
};
}  // namespace qvf
