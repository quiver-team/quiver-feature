#pragma once

#include <qvf/common.h>
#include <qvf/pipe.h>
#include <qvf/tensor_endpoint.h>

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
  int64_t* local_offsets;
  int64_t* remote_offsets;
  int64_t size;

 public:
  CollectionTask() {}
  CollectionTask(void* base_address,
                 int64_t* local_offsets,
                 int64_t* remote_offsets,
                 int64_t size,
                 int collect_from)
      : base_address(base_address),
        local_offsets(local_offsets),
        remote_offsets(remote_offsets),
        size(size),
        collect_from(collect_from) {}
};
class DistTensorClient {
 public:
  std::vector<Pipe*> pipes;
  std::vector<ComEndPoint> com_endpoints;

  // About communication
  PipeParam pipe_param;
  int server_size;
  int server_rank;

  // About IB
  infinity::core::Context* context;
  infinity::queues::QueuePairFactory* qpFactory;

  infinity::memory::Buffer* tensor_buffer;
  infinity::memory::RegionToken* tensor_token;

  // about feature client
  std::deque<CollectionTask> task_queue;

 public:
  DistTensorClient(int server_rank,
                   std::vector<ComEndPoint> com_endpoints,
                   PipeParam pipe_param) {
    this->server_rank = server_rank;
    this->com_endpoints = com_endpoints;
    this->pipe_param = pipe_param;
    server_size = com_endpoints.size();
    init_connection();
  }

  void init_connection() {
    context = new infinity::core::Context();
    qpFactory = new infinity::queues::QueuePairFactory(context);
    pipes.resize(server_size);
    for (int idx = 0; idx < server_size; idx++) {
      if (com_endpoints[idx].get_rank() == server_rank) {
        continue;
      }
      pipes[com_endpoints[idx].get_rank()] =
          new Pipe(context, qpFactory, com_endpoints[idx], pipe_param);
      pipes[com_endpoints[idx].get_rank()]->connect();
    }
  }

  torch::Tensor create_registered_float32_tensor(
      std::vector<int64_t> tensor_shape) {
    QUIVER_FEATURE_ASSERT(tensor_shape.size() == 2,
                          "Only support 2-dimensional tensor");
    auto tensor_option = torch::TensorOptions().dtype(torch::kFloat32);
    uint64_t size_in_bytes = 4;
    for (int index = 0; index < tensor_shape.size(); index++) {
      size_in_bytes *= tensor_shape[index];
    }
    tensor_buffer = new infinity::memory::Buffer(context, size_in_bytes);
    tensor_token = tensor_buffer->createRegionToken();
    return torch::from_blob(tensor_buffer->getData(),
                            {tensor_shape[0], tensor_shape[1]}, tensor_option);
  }

  torch::Tensor create_registered_float32_tensor_cuda(
      std::vector<int64_t> tensor_shape,
      int device) {
    QUIVER_FEATURE_ASSERT(tensor_shape.size() == 2,
                          "Only support 2-dimensional tensor");
    uint64_t size_in_bytes = 4;
    for (int index = 0; index < tensor_shape.size(); index++) {
      size_in_bytes *= tensor_shape[index];
    }
    tensor_buffer =
        new infinity::memory::Buffer(context, size_in_bytes, device);
    tensor_token = tensor_buffer->createRegionToken();
    auto tensor_option = torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(torch::kCUDA, device);
    return torch::from_blob(tensor_buffer->getData(),
                            {tensor_shape[0], tensor_shape[1]}, tensor_option);
  }

  void sync_read(int server_rank,
                 torch::Tensor& res_tensor,
                 torch::Tensor& local_offsets,
                 torch::Tensor& remote_offsets) {
    QUIVER_FEATURE_ASSERT(
        reinterpret_cast<uint64_t>(res_tensor.data_ptr<float>()) ==
            tensor_buffer->getAddress(),
        "Result Tensor is not created from registered buffer");

    pipes[server_rank]->read(tensor_buffer, local_offsets, remote_offsets,
                             res_tensor.size(1) * 4);
  }

  void collect_inner(CollectionTask collection_task) {
    task_queue.push_back(collection_task);
  }

  void start_feature_client() {}
};
}  // namespace qvf
