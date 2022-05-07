#pragma once

#include <qvf/com_endpoint.h>
#include <qvf/range.h>

#include <infinity/core/Context.h>
#include <infinity/memory/Buffer.h>
#include <infinity/memory/RegionToken.h>
#include <infinity/queues/QueuePair.h>
#include <infinity/queues/QueuePairFactory.h>
#include <infinity/requests/RequestToken.h>

#include <iostream>
#include <vector>

namespace qvf {

// Pipe are used for single side RDMA read to remote data servers
struct PipeParam {
  int qp_num;
  int ctx_poll_batch;
  int tx_depth;
  int post_list_size;
  int cq_mode;
  PipeParam() {}
  PipeParam(int qp_num,
            int cq_mode,
            int ctx_poll_batch,
            int tx_depth,
            int post_list_size) {
    this->qp_num = qp_num;
    this->cq_mode = cq_mode;
    this->ctx_poll_batch = ctx_poll_batch;
    this->tx_depth = tx_depth;
    this->post_list_size = post_list_size;
  }
  void set_params(int qp_num,
                  int cq_mode,
                  int ctx_poll_batch,
                  int tx_depth,
                  int post_list_size) {
    this->qp_num = qp_num;
    this->cq_mode = cq_mode;
    this->ctx_poll_batch = ctx_poll_batch;
    this->tx_depth = tx_depth;
    this->post_list_size = post_list_size;
  }
};

class Pipe {
 private:
  ComEndPoint remote_end;
  PipeParam pipe_param;
  infinity::memory::RegionToken* remote_buffer_token;
  std::vector<infinity::queues::QueuePair*> qps;
  std::vector<infinity::requests::RequestToken*> requests;
  infinity::queues::SendRequestBuffer send_buffer;
  infinity::core::Context* context;
  infinity::queues::IbvWcBuffer wc_buffer;
  int requests_size;

 public:
  Pipe(infinity::core::Context* context,
       ComEndPoint com_endpoint,
       PipeParam pipe_param) {
    this->context = context;
    this->remote_end = com_endpoint;
    this->pipe_param = pipe_param;
    qps.resize(pipe_param.qp_num);
    requests_size =
        pipe_param.tx_depth / pipe_param.cq_mode / pipe_param.post_list_size;
    requests.resize(requests_size);
    send_buffer.resize(pipe_param.post_list_size);
    wc_buffer.resize(pipe_param.ctx_poll_batch);
  }

  void connect() {
    infinity::queues::QueuePairFactory* qpFactory =
        new infinity::queues::QueuePairFactory(context);
    for (int qp_index = 0; qp_index < pipe_param.qp_num; qp_index++) {
      qps[qp_index] = qpFactory->connectToRemoteHost(
          remote_end.get_address().c_str(), remote_end.get_port());
    }
    remote_buffer_token = (infinity::memory::RegionToken*)qps[0]->getUserData();
    for (int request_index = 0; request_index < requests.size();
         request_index++) {
      requests[request_index] = new infinity::requests::RequestToken(context);
    }
  }

  void read(infinity::memory::Buffer* local_buffer,
            std::vector<uint64_t> local_offsets,
            std::vector<uint64_t> remote_offsets,
            uint64_t stride) {
    uint64_t post_list_cnt =
        (local_offsets.size() + pipe_param.post_list_size - 1) /
        pipe_param.post_list_size;
    // std::cout<<"Check Post_List_Count " << post_list_cnt << std::endl;
    // std::cout<<"Check Local_Offset_Size " << local_offsets.size() << " Check
    // Local_Offset_Size "<< remote_offsets.size()<<std::endl;

    int epoch_scnt = 0;
    for (uint64_t post_index = 0; post_index < post_list_cnt; post_index++) {
      int batch_read_size = (post_index == post_list_cnt - 1)
                                ? (local_offsets.size() -
                                   (pipe_param.post_list_size * post_index))
                                : pipe_param.post_list_size;
      // std::cout<<"Check Batch_Read_Size " << batch_read_size << std::endl;
      // std::cout<<"Check Current Index " << pipe_param.post_list_size *
      // post_index <<" Total Size " << local_offsets.size()<<std::endl;
      if (post_index == post_list_cnt - 1 ||
          post_index % pipe_param.cq_mode == (pipe_param.cq_mode - 1)) {
        qps[post_index % pipe_param.qp_num]->multiRead(
            batch_read_size, local_buffer,
            &local_offsets[post_index * pipe_param.post_list_size],
            remote_buffer_token,
            &remote_offsets[post_index * pipe_param.post_list_size], stride,
            infinity::queues::OperationFlags(), requests[epoch_scnt],
            send_buffer);
        epoch_scnt += 1;
      } else {
        qps[post_index % pipe_param.qp_num]->multiRead(
            batch_read_size, local_buffer,
            &local_offsets[post_index * pipe_param.post_list_size],
            remote_buffer_token,
            &remote_offsets[post_index * pipe_param.post_list_size], stride,
            infinity::queues::OperationFlags(), nullptr, send_buffer);
      }
      if (epoch_scnt == requests_size || post_index == post_list_cnt - 1) {
        epoch_scnt = 0;
        context->batchPollSendCompletionQueue(pipe_param.ctx_poll_batch,
                                              epoch_scnt, wc_buffer.ptr());
      }
    }
  }
};
}  // namespace qvf
