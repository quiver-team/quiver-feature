/**
 * Examples - Read/Write/Send Operations
 *
 * (c) 2018 Claude Barthels, ETH Zurich
 * Contact: claudeb@inf.ethz.ch
 *
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cassert>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include <infinity/core/Context.h>
#include <infinity/memory/Buffer.h>
#include <infinity/memory/RegionToken.h>
#include <infinity/queues/QueuePair.h>
#include <infinity/queues/QueuePairFactory.h>
#include <infinity/requests/RequestToken.h>

#include <qvf/com_endpoint.h>
#include <qvf/common.h>
#include <qvf/pipe.h>
#include <qvf/range.h>

#define PORT_NUMBER 3344
#define SERVER_IP "155.198.152.17"

#define NODE_COUNT 120000LL
#define FEATURE_DIM 256LL
#define FEATURE_TYPE_SIZE 4LL
#define TEST_COUNT 8192LL
#define ITER_NUM 10LL
#define POST_LIST_SIZE 16LL
#define CQ_MOD 16LL
#define QP_NUM 2LL
#define TX_DEPTH 2048LL
#define CTX_POLL_BATCH 16LL

int min(int a, int b) {
  if (a < b) {
    return a;
  }
  return b;
}

uint64_t timeDiff(struct timeval stop, struct timeval start) {
  return (stop.tv_sec * 1000000L + stop.tv_usec) -
         (start.tv_sec * 1000000L + start.tv_usec);
}

float* allocate_float_feature(bool set_value);

bool mem_check(float* data_buffer) {
  float index = 0;
  bool have_valid_data = false;
  for (u_int64_t start = 0; start < NODE_COUNT; start += 1) {
    for (int dim = 0; dim < FEATURE_DIM; dim++) {
      if (data_buffer[start * FEATURE_DIM + dim] != 0) {
        have_valid_data = true;
      }
    }
  }
  QUIVER_FEATURE_ASSERT(have_valid_data == true, "No valid data is copied")

  for (u_int64_t start = 0; start < NODE_COUNT; start += 1) {
    float expected_value =
        (data_buffer[start * FEATURE_DIM] == 0) ? 0 : float(start);
    std::cout << data_buffer[start * FEATURE_DIM] << " ";
    for (u_int64_t dim = 0; dim < FEATURE_DIM; dim++) {
      QUIVER_FEATURE_ASSERT(
          data_buffer[start * FEATURE_DIM + dim] == expected_value,
          "Result Check Failed At (%lld, %lld)!, Expected %f, Got %f\n", start,
          dim, expected_value, data_buffer[start * FEATURE_DIM + dim]);
    }
  }
  return true;
}

void test_pipe(int argc, char** argv) {
  bool random = true;
  bool sort_index = false;

  while (argc > 1) {
    if (argv[1][0] == '-') {
      switch (argv[1][1]) {
        case 'l': {
          random = false;
          break;
        }
        case 't': {
          sort_index = true;
          break;
        }
      }
    }
    ++argv;
    --argc;
  }
  if (random) {
    printf("Test Random Data Access \n");
  } else {
    printf("Test Sequential Data Access \n");
  }
  if (sort_index) {
    printf("Test Data Access With TLB Optimization\n");
  }

  std::vector<infinity::queues::QueuePair*> qps;
  infinity::core::Context* context = new infinity::core::Context();
  infinity::queues::QueuePairFactory* qpFactory =
      new infinity::queues::QueuePairFactory(context);

  qps.resize(QP_NUM);
  qvf::ComEndPoint endpoint(0, SERVER_IP, PORT_NUMBER);
  qvf::PipeParam pipe_param(QP_NUM, CTX_POLL_BATCH, TX_DEPTH,
                            POST_LIST_SIZE);
  qvf::Pipe quiver_pipe(context, qpFactory, endpoint, pipe_param);
  quiver_pipe.connect();

  printf("Creating buffers\n");
  std::vector<infinity::memory::Buffer*> buffers;
  float* client_data_buffer = allocate_float_feature(false);
  infinity::memory::Buffer* buffer1Sided = new infinity::memory::Buffer(
      context, client_data_buffer,
      NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE);
  infinity::memory::Buffer* buffer2Sided =
      new infinity::memory::Buffer(context, 128 * sizeof(char));

  printf("Reading content from remote buffer\n");
  infinity::requests::RequestToken requestToken(context);

  printf("Start Real Test \n");
  // auto start = std::chrono::system_clock::now();
  struct timeval start, stop;
  uint64_t time_consumed = 0;
  std::vector<int64_t> local_offsets(TEST_COUNT * POST_LIST_SIZE);
  std::vector<int64_t> remote_offsets(TEST_COUNT * POST_LIST_SIZE);
  if (sort_index) {
    for (int iter_index = 0; iter_index < ITER_NUM; iter_index++) {
      std::vector<int> all_request_nodes(TEST_COUNT * POST_LIST_SIZE);
      for (int i = 0; i < TEST_COUNT * POST_LIST_SIZE; i++) {
        all_request_nodes[i] = rand() % NODE_COUNT;
      }
      std::sort(all_request_nodes.begin(), all_request_nodes.end());
      for (int i = 0; i < TEST_COUNT * POST_LIST_SIZE; i++) {
        uint64_t remote_node_offset =
            all_request_nodes[i] * FEATURE_DIM * FEATURE_TYPE_SIZE;
        local_offsets[i] = remote_node_offset;
        remote_offsets[i] = remote_node_offset;
      }
      gettimeofday(&start, NULL);

      quiver_pipe.read(buffer1Sided, local_offsets, remote_offsets,
                       FEATURE_DIM * FEATURE_TYPE_SIZE);
      gettimeofday(&stop, NULL);
      time_consumed += timeDiff(stop, start);
    }
  } else {
    for (int iter_index = 0; iter_index < ITER_NUM; iter_index++) {
      for (int k = 0; k < TEST_COUNT * POST_LIST_SIZE; k++) {
        int request_node = k % NODE_COUNT;
        if (random) {
          request_node = rand() % NODE_COUNT;
        }
        uint64_t remote_node_offset =
            request_node * FEATURE_DIM * FEATURE_TYPE_SIZE;
        local_offsets[k] = remote_node_offset;
        remote_offsets[k] = remote_node_offset;
      }
      gettimeofday(&start, NULL);
      quiver_pipe.read(buffer1Sided, local_offsets, remote_offsets,
                       FEATURE_DIM * FEATURE_TYPE_SIZE);
      gettimeofday(&stop, NULL);
      time_consumed += timeDiff(stop, start);
    }
  }

  printf("Avg Bandwidth is %f MB/s\n",
         (POST_LIST_SIZE * TEST_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE *
          ITER_NUM / (1024.0 * 1024.0)) /
             (((double)time_consumed) / 1000000L));

  printf("Memory checking..., Please wait...\n");
  if (!mem_check(client_data_buffer)) {
    fprintf(stderr, "Memory Check Failed, Benchmark Failed!\n");
  } else {
    printf("Memory check success! Congrats!\n");
  }

  delete buffer1Sided;
  delete buffer2Sided;

  for (int index = 0; index < QP_NUM; index++) {
    delete qps[index];
  }
  delete qpFactory;
  delete context;
}
