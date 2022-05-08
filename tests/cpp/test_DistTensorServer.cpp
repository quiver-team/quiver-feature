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
#define QP_NUM 16LL
#define TX_DEPTH 2048LL
#define CTX_POLL_BATCH 16LL

uint8_t* allocate_feature(bool set_value);

void test_dist_tensor_server(int argc, char** argv) {
  qvf::PipeParam pipe_param(QP_NUM, CQ_MOD, CTX_POLL_BATCH, TX_DEPTH,
                            POST_LIST_SIZE);
  qvf::DistTensorServer dist_tensor_server(PORT_NUMBER, 1, 1);
  uint8_t* server_data_buffer = allocate_feature(true);
  dist_tensor_server.serve(server_data_buffer,
                           NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE);
}
