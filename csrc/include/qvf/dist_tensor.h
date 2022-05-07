#pragma once
#include <qvf/com_endpoint.h>
#include <qvf/range.h>
#include <vector>

namespace qvf {
class DistTensor {
 private:
  void* local_data_buffer;
  uint64_t stride;
  Range local_range;
  ComEndPoint local_end_com_point;
};
}  // namespace qvf
