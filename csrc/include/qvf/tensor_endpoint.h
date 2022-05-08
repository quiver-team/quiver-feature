#pragma once
#include <qvf/com_endpoint.h>
#include <qvf/range.h>
namespace qvf {
class TensorEndPoint {
 public:
  ComEndPoint com_endpoint;
  Range range;

 public:
  TensorEndPoint(ComEndPoint com_endpoint, Range range) {
    this->com_endpoint = com_endpoint;
    this->range = range;
  }

  TensorEndPoint(std::string ip,
                 int port,
                 int rank,
                 int64_t range_start,
                 int64_t range_end) {
    this->com_endpoint = ComEndPoint(rank, ip, port);
    this->range = Range(range_start, range_end);
  }

  TensorEndPoint(std::string ip, int port, int rank, Range range) {
    this->com_endpoint = ComEndPoint(rank, ip, port);
    this->range = range;
  }

  TensorEndPoint& operator=(const TensorEndPoint& other) {
    this->com_endpoint = other.com_endpoint;
    this->range = other.range;
    return *this;
  }
};
}  // namespace qvf
