#pragma once

#include <qvf/com_endpoint.h>
#include <qvf/pipe.h>
#include <qvf/range.h>

#include <infinity/core/Context.h>
#include <infinity/memory/Buffer.h>
#include <infinity/memory/RegionToken.h>
#include <infinity/queues/QueuePair.h>
#include <infinity/queues/QueuePairFactory.h>
#include <infinity/requests/RequestToken.h>
#include <torch/extension.h>
#include <chrono>
#include <deque>
#include <thread>
#include <vector>

namespace qvf {
struct CollectionTask {
 public:
  void* base_address;
  int collect_from;
  uint64_t* local_offsets;
  uint64_t* remote_offsets;
  uint64_t size;

 public:
  CollectionTask() {}
  CollectionTask(void* base_address,
                 uint64_t* local_offsets,
                 uint64_t* remote_offsets,
                 uint64_t size,
                 int collect_from)
      : base_address(base_address),
        local_offsets(local_offsets),
        remote_offsets(remote_offsets),
        size(size),
        collect_from(collect_from) {}
};
class DistTensor {
 private:
  std::vector<Pipe> pipes;
  std::vector<Range> ranges;
  std::vector<ComEndPoint> com_endpoints;
  void* feature_data;

  uint64_t stride_in_bytes;
  Range local_range;
  uint64_t size_in_bytes;

  // About communication
  ComEndPoint local_endpoint;
  PipeParam pipe_param;
  int world_size;
  int rank;

  // About IB
  infinity::core::Context* context;
  infinity::queues::QueuePairFactory* qpFactory;

  infinity::memory::Buffer* feature_buffer;
  infinity::memory::Buffer* tensor_buffer;

  infinity::memory::RegionToken* feature_token;
  infinity::memory::RegionToken* tensor_token;

  // about feature server
  std::thread server_thread;

  // about feature client
  std::deque<CollectionTask> task_queue;

 public:
  DistTensor(void* data, Range local_range, uint64_t stride_in_bytes)
      : feature_data(data),
        local_range(local_range),
        stride_in_bytes(stride_in_bytes) {
    size_in_bytes =
        (local_range.range_end() - local_range.range_start()) * stride_in_bytes;
  }

  void init_comm(int world_size,
                 ComEndPoint local_endpoint,
                 PipeParam pipe_param) {
    this->local_endpoint = local_endpoint;
    this->world_size = world_size;
    this->rank = local_endpoint.get_rank();

    pipes.resize(world_size);
    ranges.resize(world_size);
    com_endpoints.resize(world_size);
    ranges[rank] = local_range;
    com_endpoints[rank] = local_endpoint;

    // start feature server
    context = new infinity::core::Context();
    qpFactory = new infinity::queues::QueuePairFactory(context);
  }
  void connect(ComEndPoint remote_endpoint, Range remote_range) {
    ranges[remote_endpoint.get_rank()] = remote_range;
    com_endpoints[remote_endpoint.get_rank()] = remote_endpoint;
    pipes[remote_endpoint.get_rank()] =
        Pipe(context, remote_endpoint, pipe_param);
    pipes[remote_endpoint.get_rank()].connect();
  }

  void collect_inner(CollectionTask collection_task) {
    task_queue.push_back(collection_task);
  }

  void collect(int collect_from,
               torch::Tensor& res_tensor,
               torch::Tensor& local_offsets,
               torch::Tensor& remote_offsets) {
    //
  }

  void start_feature_client() {}
};
}  // namespace qvf
