
#pragma once

#include <qvf/com_endpoint.h>

#include <infinity/core/Context.h>
#include <infinity/memory/Buffer.h>
#include <infinity/memory/RegionToken.h>
#include <infinity/queues/QueuePair.h>
#include <infinity/queues/QueuePairFactory.h>
#include <infinity/requests/RequestToken.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include <torch/extension.h>

namespace qvf {
class DistTensorServer {
 private:
  int port;
  int world_size;
  int qp_per_pipe;

  infinity::core::Context* context;
  infinity::queues::QueuePairFactory* qpFactory;
  infinity::memory::Buffer* feature_buffer;
  infinity::memory::RegionToken* bufferToken;

  std::thread server_thread;

 public:
  DistTensorServer(int port, int world_size, int qp_per_pipe)
      : port(port), world_size(world_size), qp_per_pipe(qp_per_pipe) {
    context = new infinity::core::Context();
    qpFactory = new infinity::queues::QueuePairFactory(context);
    qpFactory->bindToPort(port);
  }

  void join() { server_thread.join(); }

  void serve(void* data, int64_t size_in_bytes) {
    feature_buffer =
        new infinity::memory::Buffer(context, data, (uint64_t)size_in_bytes);
    bufferToken = feature_buffer->createRegionToken();
    server_thread =
        std::thread(run, qpFactory, bufferToken, qp_per_pipe * world_size);
  }

  void serve_tensor(torch::Tensor& data) {
    std::cout << "Registering Buffer, Please Wait..." << std::endl;
    uint64_t size_in_bytes = data.numel() * 4;
    feature_buffer = new infinity::memory::Buffer(
        context, data.data_ptr<float>(), size_in_bytes);
    bufferToken = feature_buffer->createRegionToken();
    server_thread = std::thread(run, qpFactory, bufferToken,
                                qp_per_pipe * (world_size - 1));
  }

  static void run(infinity::queues::QueuePairFactory* qpFactory,
                  infinity::memory::RegionToken* bufferToken,
                  int total_qp_num) {
    std::cout << "Buffer Registeration Done! Ready To Receive Connections, "
                 "Start Your Clients Now"
              << std::endl;
    for (int qp_index = 0; qp_index < total_qp_num; qp_index++) {
      qpFactory->acceptIncomingConnection(
          bufferToken, sizeof(infinity::memory::RegionToken));
    }

    while (1) {
      std::this_thread::sleep_for(std::chrono::seconds(10));  // 10s
    }
  }
};

}  // namespace qvf
