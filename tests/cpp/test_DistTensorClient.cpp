#include <qvf/common.h>
#include <qvf/dist_tensor_client.h>
#include <qvf/pipe.h>
#include <qvf/tensor_endpoint.h>

#include <iostream>
#include <vector>

#include <torch/extension.h>

#define PORT_NUMBER 3344
#define SERVER_IP "155.198.152.17"

#define NODE_COUNT 120000LL
#define FEATURE_DIM 256LL
#define FEATURE_TYPE_SIZE 4LL
#define SAMPLE_NUM 80000LL
#define TEST_COUNT 8192LL
#define ITER_NUM 10LL
#define POST_LIST_SIZE 16LL
#define CQ_MOD 16LL
#define QP_NUM 1LL
#define TX_DEPTH 2048LL
#define CTX_POLL_BATCH 16LL

void check_tensor_res(torch::Tensor& res_tensor,
                      torch::Tensor& remote_offsets) {
  float* res = res_tensor.data_ptr<float>();
  int stride = res_tensor.size(1);
  int64_t* offsets = remote_offsets.data_ptr<int64_t>();
  for (int row = 0; row < remote_offsets.size(0); row++) {
    for (int col = 0; col < res_tensor.size(1); col++) {
      float expected_value =
          float(offsets[row]) / FEATURE_DIM * FEATURE_TYPE_SIZE;
      QUIVER_FEATURE_ASSERT(
          res[row * stride + col] == expected_value,
          "Result Check Failed At (%d, %d)!, Expected %f, Got %f\n", row, col,
          expected_value, res[row * stride + col]);
    }
  }
  printf("Result Check Passed!\n");
}

void test_dist_tensor_client(int argc, char** argv) {
  qvf::PipeParam pipe_param(QP_NUM, CQ_MOD, CTX_POLL_BATCH, TX_DEPTH,
                            POST_LIST_SIZE);

  qvf::TensorEndPoint local_tensor_end_point(
      qvf::ComEndPoint(0, SERVER_IP, PORT_NUMBER), qvf::Range(0, NODE_COUNT));
  qvf::TensorEndPoint remote_tensor_end_point(
      qvf::ComEndPoint(1, SERVER_IP, PORT_NUMBER), qvf::Range(0, NODE_COUNT));
  std::vector<qvf::TensorEndPoint> tensor_endpoints{local_tensor_end_point,
                                                    remote_tensor_end_point};
  qvf::DistTensorClient dist_tensor_client(0, tensor_endpoints, pipe_param);
  std::vector<int64_t> shape{SAMPLE_NUM, FEATURE_DIM};

  torch::Tensor registered_tensor =
      dist_tensor_client.create_registered_float32_tensor(shape);
  std::vector<int64_t> local_offsets(SAMPLE_NUM);
  std::vector<int64_t> remote_offsets(SAMPLE_NUM);

  for (int index = 0; index < SAMPLE_NUM; index++) {
    local_offsets[index] = index * FEATURE_DIM * FEATURE_TYPE_SIZE;
    remote_offsets[index] =
        rand() % NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE;
  }

  for (int index = 0; index < 10; index++) {
    std::cout << remote_offsets[index] << " ";
  }
  std::cout << std::endl;

  auto tensor_option = torch::TensorOptions().dtype(torch::kInt64);
  torch::Tensor local_offsets_tensor =
      torch::from_blob(&local_offsets[0], {SAMPLE_NUM}, tensor_option);
  torch::Tensor remote_offsets_tensor =
      torch::from_blob(&remote_offsets[0], {SAMPLE_NUM}, tensor_option);

  dist_tensor_client.sync_read(1, registered_tensor, local_offsets_tensor,
                               remote_offsets_tensor);
  check_tensor_res(registered_tensor, remote_offsets_tensor);
}
