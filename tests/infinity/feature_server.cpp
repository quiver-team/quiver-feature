/**
 * Examples - Read/Write/Send Operations
 *
 * (c) 2018 Claude Barthels, ETH Zurich
 * Contact: claudeb@inf.ethz.ch
 *
 */

#include <cassert>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>

#include <infinity/core/Context.h>
#include <infinity/memory/Buffer.h>
#include <infinity/memory/RegionToken.h>
#include <infinity/queues/QueuePair.h>
#include <infinity/queues/QueuePairFactory.h>
#include <infinity/requests/RequestToken.h>

#define PORT_NUMBER 3344
#define SERVER_IP "155.198.152.17"

#define NODE_COUNT 120000000LL
#define FEATURE_DIM 256LL
#define FEATURE_TYPE_SIZE 4LL
#define TEST_COUNT 8192LL
#define ITER_NUM 10LL
#define POST_LIST_SIZE 16LL
#define CQ_MOD 16LL
#define QP_NUM 16LL
#define TX_DEPTH 2048LL
#define CTX_POLL_BATCH 16LL
int min(int a, int b){
    if(a < b){
        return a;
    }
    return b;
}


uint64_t timeDiff(struct timeval stop, struct timeval start) {
  return (stop.tv_sec * 1000000L + stop.tv_usec) -
         (start.tv_sec * 1000000L + start.tv_usec);
}


uint8_t* allocate_feature(bool set_value){
  uint8_t* buffer = (uint8_t*) malloc(NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE);
  if(set_value){
    int index = 0;
    for(u_int64_t start=0; start < NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE; start += FEATURE_DIM * FEATURE_TYPE_SIZE){
      for(int dim = 0; dim < FEATURE_DIM; dim ++){
        *((int*)(buffer + start + dim * FEATURE_TYPE_SIZE)) = index;
      }
      index += 1;
    }
  }else{
    memset(buffer, 0, NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE);
  }
  return buffer;
}
class EndPoint{

};
class Range{

};

class DistTensor{

    public:
    DistTensor(void* buffer, uint64_t size, int stride){}
    void append(EndPoint endpoint, Range range){}
    
};

bool mem_check(uint8_t* data_buffer){
    int index = 0;
    bool have_valid_data = false; 
    for(u_int64_t start=0; start < NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE; start += FEATURE_DIM * FEATURE_TYPE_SIZE){
      for(int dim = 0; dim < FEATURE_DIM; dim ++){
        if (*((int*)(data_buffer + start + dim * FEATURE_TYPE_SIZE)) != 0){
          have_valid_data = true;
        }
      }
    }
    if(!have_valid_data){
      fprintf(stderr, "You know, actually no data is copied successfully");
    }

    for(u_int64_t start=0; start < NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE; start += FEATURE_DIM * FEATURE_TYPE_SIZE){
      for(int dim = 0; dim < FEATURE_DIM; dim ++){
        //std::cout<< *((int*)(data_buffer + start + dim * FEATURE_TYPE_SIZE))<< " ";
        if (*((int*)(data_buffer + start + dim * FEATURE_TYPE_SIZE)) != 0 && *((int*)(data_buffer + start + dim * FEATURE_TYPE_SIZE)) != index){
          fprintf(stderr, "At %lld: Epected %d, But got %d\n", start / FEATURE_DIM * FEATURE_TYPE_SIZE , index, *((int*)(data_buffer + start + dim * FEATURE_TYPE_SIZE)));
          return false;
        }
      }
      //std::cout<<std::endl;
      index += 1;
    }
    return have_valid_data & true;
}

// Usage: ./progam -s for server and ./program for client component
int main(int argc, char **argv) {

  bool isServer = false;
  bool random = true;
  bool sort_index = false;

  while (argc > 1) {
    if (argv[1][0] == '-') {
      switch (argv[1][1]) {
        case 's': {
          isServer = true;
          break;
        }
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
  if(random){
    printf("Test Random Data Access \n");
  }else{
    printf("Test Sequential Data Access \n");
  }
  if(sort_index){
    printf("Test Data Access With TLB Optimization\n");
  }
  std::vector<infinity::queues::QueuePair*> qps;
  infinity::core::Context *context = new infinity::core::Context();
  infinity::queues::QueuePairFactory *qpFactory =
      new infinity::queues::QueuePairFactory(context);

  qps.resize(QP_NUM);

  if (isServer) {

    printf("Creating buffers to read from and write to\n");
    std::cout << "Server Buffer Size " << NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE << std::endl;
    uint8_t* server_data_buffer = allocate_feature(true);
    infinity::memory::Buffer *bufferToReadWrite =
        new infinity::memory::Buffer(context, server_data_buffer, NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE);
    infinity::memory::RegionToken *bufferToken =
        bufferToReadWrite->createRegionToken();
    
    printf("Creating buffers to receive a message\n");
		infinity::memory::Buffer *bufferToReceive = new infinity::memory::Buffer(context, 128 * sizeof(char));
		context->postReceiveBuffer(bufferToReceive);


    printf("Setting up connection (blocking)\n");
    qpFactory->bindToPort(PORT_NUMBER);
    for(int qp_index=0; qp_index < QP_NUM; qp_index++){
        qps[qp_index] = qpFactory->acceptIncomingConnection(bufferToken, sizeof(infinity::memory::RegionToken));
    }
    printf("Waiting for message (blocking)\n");
    infinity::core::receive_element_t receiveElement;
    while (!context->receive(&receiveElement))
        ;

    printf("Message received\n");
    delete bufferToReadWrite;
    delete bufferToReceive;

  } else {

    std::vector<uint64_t> local_offsets(POST_LIST_SIZE, 0);
    std::vector<uint64_t> remote_offsets(POST_LIST_SIZE, 0);
    infinity::queues::SendRequestBuffer send_buffer(POST_LIST_SIZE);
    infinity::queues::IbvWcBuffer wc_buffer = infinity::queues::IbvWcBuffer(CTX_POLL_BATCH);
    int epoch_scnt = 0;

    printf("Connecting to remote node\n");
    std::vector<infinity::memory::RegionToken *> remoteBufferTokens;
    remoteBufferTokens.resize(QP_NUM);
    for(int qp_index = 0; qp_index < QP_NUM; qp_index ++){
        qps[qp_index] = qpFactory->connectToRemoteHost(SERVER_IP, PORT_NUMBER);
        remoteBufferTokens[qp_index] = (infinity::memory::RegionToken *)qps[qp_index]->getUserData();
    }

    int REQUEST_BUFFER_SIZE = TX_DEPTH / CQ_MOD / POST_LIST_SIZE;
    std::vector<infinity::requests::RequestToken *> requests;
    for (int i = 0; i < REQUEST_BUFFER_SIZE; i++) {
      requests.push_back(new infinity::requests::RequestToken(context));
    }


    printf("Creating buffers\n");
    std::vector<infinity::memory::Buffer *> buffers;
    uint8_t* client_data_buffer = allocate_feature(false);
    infinity::memory::Buffer *buffer1Sided = new infinity::memory::Buffer(context, client_data_buffer, NODE_COUNT * FEATURE_DIM * FEATURE_TYPE_SIZE);
    infinity::memory::Buffer *buffer2Sided = new infinity::memory::Buffer(context, 128 * sizeof(char));


    printf("Reading content from remote buffer\n");
    infinity::requests::RequestToken requestToken(context);

    // warm up

    printf("Warm up\n");
    for (int k = 0; k < 10; k++) {
      int request_node = rand() % NODE_COUNT;
      uint64_t offset = request_node * FEATURE_DIM * FEATURE_TYPE_SIZE;
      //std::cout << "Getting Data From " << offset << " To " << offset + FEATURE_DIM * FEATURE_TYPE_SIZE << std::endl;
      qps[k % QP_NUM]->read(buffer1Sided, offset, remoteBufferTokens[k % QP_NUM], offset, FEATURE_DIM * FEATURE_TYPE_SIZE,
                infinity::queues::OperationFlags(), requests[k % REQUEST_BUFFER_SIZE]);
      requests[k % REQUEST_BUFFER_SIZE]->waitUntilCompleted();
    }

    printf("Start Real Test \n");
    //auto start = std::chrono::system_clock::now();
    struct timeval start, stop;
    uint64_t time_consumed = 0;
    if(sort_index){
      for(int iter_index = 0; iter_index < ITER_NUM; iter_index ++){
        std::vector<int> all_request_nodes(TEST_COUNT * POST_LIST_SIZE);
        for(int i=0; i < TEST_COUNT * POST_LIST_SIZE; i ++){
            all_request_nodes[i] = rand() % NODE_COUNT;
        }
        std::sort(all_request_nodes.begin(), all_request_nodes.end());
        gettimeofday(&start, NULL);

        for (int k = 0; k < TEST_COUNT; k++) {
          for(int multi_read_index = 0; multi_read_index < POST_LIST_SIZE; multi_read_index ++){
              uint64_t remote_node_offset = all_request_nodes[k * POST_LIST_SIZE + multi_read_index] * FEATURE_DIM * FEATURE_TYPE_SIZE;
              local_offsets[multi_read_index] = remote_node_offset;
              remote_offsets[multi_read_index] = remote_node_offset;
          }

          if(k % CQ_MOD ==  CQ_MOD -1){
              qps[k % QP_NUM]->multiRead(buffer1Sided, local_offsets, remoteBufferTokens[k % QP_NUM], remote_offsets, FEATURE_DIM * FEATURE_TYPE_SIZE,
                          infinity::queues::OperationFlags(), requests[epoch_scnt], send_buffer);
              epoch_scnt += 1;
              
          }else{
              qps[k % QP_NUM]->multiRead(buffer1Sided, local_offsets, remoteBufferTokens[k % QP_NUM], remote_offsets, FEATURE_DIM * FEATURE_TYPE_SIZE,
                          infinity::queues::OperationFlags(), nullptr, send_buffer);
          }
          if(epoch_scnt ==  REQUEST_BUFFER_SIZE){
            epoch_scnt = 0;
            context->batchPollSendCompletionQueue(CTX_POLL_BATCH, REQUEST_BUFFER_SIZE, wc_buffer.ptr());
          }
        }
        gettimeofday(&stop, NULL);
        time_consumed += timeDiff(stop, start);
      }
      
    }
    else{
      gettimeofday(&start, NULL);
      for(int iter_index = 0; iter_index < ITER_NUM; iter_index++){
        for (int k = 0; k < TEST_COUNT; k++) {
            for(int multi_read_index = 0; multi_read_index < POST_LIST_SIZE; multi_read_index ++){
                int request_node = (k + multi_read_index) % NODE_COUNT;
                if(random){
                    request_node = rand() % NODE_COUNT;
                }
                uint64_t remote_node_offset = request_node * FEATURE_DIM * FEATURE_TYPE_SIZE;
                local_offsets[multi_read_index] = remote_node_offset;
                remote_offsets[multi_read_index] = remote_node_offset;
            }

            if(k % CQ_MOD ==  CQ_MOD -1){
                qps[k % QP_NUM]->multiRead(buffer1Sided, local_offsets, remoteBufferTokens[k % QP_NUM], remote_offsets, FEATURE_DIM * FEATURE_TYPE_SIZE,
                            infinity::queues::OperationFlags(), requests[epoch_scnt], send_buffer);
                epoch_scnt += 1;

            }else{
                qps[k % QP_NUM]->multiRead(buffer1Sided, local_offsets, remoteBufferTokens[k % QP_NUM], remote_offsets, FEATURE_DIM * FEATURE_TYPE_SIZE,
                            infinity::queues::OperationFlags(), nullptr, send_buffer);
            }
            if(epoch_scnt == REQUEST_BUFFER_SIZE){
              epoch_scnt = 0;
              context->batchPollSendCompletionQueue(CTX_POLL_BATCH, REQUEST_BUFFER_SIZE, wc_buffer.ptr());
            }
        }
      }
      gettimeofday(&stop, NULL);
      time_consumed += timeDiff(stop, start);
    }

    printf("Avg Bandwidth is %f MB/s\n", (POST_LIST_SIZE * TEST_COUNT *  FEATURE_DIM * FEATURE_TYPE_SIZE * ITER_NUM / (1024.0 * 1024.0) )/ (((double) time_consumed) / 1000000L) );

    printf("Sending message to remote host from QueuePair %lld\n", QP_NUM-1);
    qps[QP_NUM-1]->send(buffer2Sided, &requestToken);
    requestToken.waitUntilCompleted();

    printf("Memory checking..., Please wait...\n");
    if(!mem_check(client_data_buffer)){
      fprintf(stderr, "Memory Check Failed, Benchmark Failed!\n");
    }else{
      printf("Memory check success! Congrats!\n");
    }

    delete buffer1Sided;
    delete buffer2Sided;
  }

  for(int index = 0; index < QP_NUM; index++){
    delete qps[index];
  }
  delete qpFactory;
  delete context;

  return 0;
}