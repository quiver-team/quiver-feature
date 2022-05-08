#include <qvf/dist_tensor_server.h>
#include <qvf/pipe.h>

#define PORT_NUMBER 3344
#define SERVER_IP "155.198.152.17"

#define NODE_COUNT 120000LL
#define FEATURE_DIM 256LL
#define FEATURE_TYPE_SIZE 4LL
#define TEST_COUNT 8192LL
#define ITER_NUM 10LL
#define POST_LIST_SIZE 16LL
#define CQ_MOD 16LL
#define QP_NUM 1LL
#define TX_DEPTH 2048LL
#define CTX_POLL_BATCH 16LL

float* allocate_float_feature(bool set_value) {
  float* buffer = (float*)malloc(NODE_COUNT * FEATURE_DIM * sizeof(float));
  float index = 0;
  for (u_int64_t start = 0; start < NODE_COUNT; start += 1) {
    for (int dim = 0; dim < FEATURE_DIM; dim++) {
      if (set_value)
        buffer[start * FEATURE_DIM + dim] = index;
      else
        buffer[start * FEATURE_DIM + dim] = 0;
    }
    index += 1;
  }
  return buffer;
}

void test_dist_tensor_server(int argc, char** argv) {
  qvf::PipeParam pipe_param(QP_NUM, CQ_MOD, CTX_POLL_BATCH, TX_DEPTH,
                            POST_LIST_SIZE);
  qvf::DistTensorServer dist_tensor_server(PORT_NUMBER, 1, 1);
  float* server_data_buffer = allocate_float_feature(true);
  dist_tensor_server.serve(server_data_buffer,
                           NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE);
}
