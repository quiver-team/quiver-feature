#include <qvf/com_endpoint.h>
#include <qvf/common.h>
#include <qvf/dist_tensor_client.h>
#include <qvf/pipe.h>

#include <iostream>
#include <vector>

#include <torch/extension.h>

#define PORT_NUMBER 3344
#define SERVER_IP "155.198.152.17"

#define NODE_COUNT 120000LL
#define FEATURE_DIM 256LL
#define FEATURE_TYPE_SIZE 4LL
#define SAMPLE_NUM 80960LL
#define TEST_COUNT 8192LL
#define ITER_NUM 10LL
#define POST_LIST_SIZE 16LL
#define CQ_MOD 16LL
#define QP_NUM 2LL
#define TX_DEPTH 2048LL
#define CTX_POLL_BATCH 16LL

int min(int a, int b);

void print_tensor_res(torch::Tensor& res_tensor) {
  float* res = res_tensor.data_ptr<float>();
  for (int col = 0; col < res_tensor.size(1); col++) {
    std::cout << res[0 * res_tensor.size(1) + col] << " ";
  }
  std::cout << std::endl;
}
void check_tensor_res(torch::Tensor& res_tensor,
                      torch::Tensor& remote_offsets) {
  float* res = res_tensor.data_ptr<float>();
  int stride = res_tensor.size(1);
  int64_t* offsets = remote_offsets.data_ptr<int64_t>();
  for (int row = 0; row < remote_offsets.size(0); row++) {
    for (int col = 0; col < res_tensor.size(1); col++) {
      float expected_value =
          float(offsets[row]) / (FEATURE_DIM * FEATURE_TYPE_SIZE);
      QUIVER_FEATURE_ASSERT(
          res[row * stride + col] == expected_value,
          "Result Check Failed At (%d, %d)!, Expected %f, Got %f\n", row, col,
          expected_value, res[row * stride + col]);
    }
  }
  printf("Result Check Passed, Congrats!\n");
}

void test_dist_tensor_client(int argc, char** argv) {
  qvf::PipeParam pipe_param(QP_NUM, CTX_POLL_BATCH, TX_DEPTH,
                            POST_LIST_SIZE);

  qvf::ComEndPoint local_com_end_point(0, SERVER_IP, PORT_NUMBER);
  qvf::ComEndPoint remote_com_end_point(1, SERVER_IP, PORT_NUMBER);
  std::vector<qvf::ComEndPoint> com_endpoints{local_com_end_point,
                                              remote_com_end_point};
  qvf::DistTensorClient dist_tensor_client(0, com_endpoints, pipe_param);
  std::vector<int64_t> shape{SAMPLE_NUM, FEATURE_DIM};

  torch::Tensor registered_tensor =
      dist_tensor_client.create_registered_float32_tensor(shape);

  std::vector<int64_t> local_offsets(SAMPLE_NUM);
  std::vector<int64_t> remote_offsets(SAMPLE_NUM);

  for (int index = 0; index < SAMPLE_NUM; index++) {
    local_offsets[index] = index * FEATURE_DIM * FEATURE_TYPE_SIZE;
    remote_offsets[index] =
        rand() % NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE;
    // remote_offsets[index] = FEATURE_DIM * FEATURE_TYPE_SIZE;
  }

  for (int index = 0; index < min(1, SAMPLE_NUM); index++) {
    std::cout << "Collect Node "
              << remote_offsets[index] / (FEATURE_DIM * FEATURE_TYPE_SIZE)
              << ": " << local_offsets[index] << "<-" << remote_offsets[index]
              << std::endl;
  }
  std::cout << std::endl;

  auto tensor_option = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor local_offsets_tensor =
      torch::from_blob(&local_offsets[0], {SAMPLE_NUM}, tensor_option);
  torch::Tensor remote_offsets_tensor =
      torch::from_blob(&remote_offsets[0], {SAMPLE_NUM}, tensor_option);

  dist_tensor_client.sync_read(1, registered_tensor, local_offsets_tensor,
                               remote_offsets_tensor);
  // print_tensor_res(registered_tensor);
  check_tensor_res(registered_tensor, remote_offsets_tensor);
}
