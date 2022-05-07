#include <qvf/pipe.h>
#include <qvf/com_endpoint.h>
#include <qvf/range.h>
#include <torch/extension.h>
#include <vector>
#include <thread>
#include <chrono>

namespace qvf{
    class DistTensor{
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
            infinity::core::Context * context;
            infinity::queues::QueuePairFactory *qpFactory;

            infinity::memory::Buffer *feature_buffer;
            infinity::memory::Buffer *tensor_buffer;
            
            infinity::memory::RegionToken *feature_token;
            infinity::memory::RegionToken *tensor_token;

            // about feature server
            std::thread server_thread;

        public:
            DistTensor(void* data, Range local_range, uint64_t stride_in_bytes): feature_data(data), local_range(local_range), stride_in_bytes(stride_in_bytes){
                size_in_bytes = (local_range.range_end() - local_range.range_start) * stride_in_bytes;
            }

            void init_comm(int world_size, ComEndPoint local_endpoint, PipeParam pipe_param){
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
                feature_buffer = new infinity::memory::Buffer(context, feature_data, size_in_bytes);
                server_thread = std::thread(start_feature_server);
            }
            void connect(ComEndPoint remote_endpoint, Range remote_range){
                ranges[remote_endpoint.get_rank()] = remote_range;
                com_endpoints[remote_endpoint.get_rank()] = remote_endpoint;
                pipes[remote_endpoint.get_rank()] = Pipe(context, remote_endpoint, pipe_param);
                pipes[remote_endpoint.get_rank()].connect();
            }

            void start_feature_server(){
                int to_be_received = pipe_param.qp_num * world_size;
                qpFactory->bindToPort(local_endpoint.get_port());
                for(int qp_index=0; qp_index < pipe_param.qp_num * world_size; qp_index++){
                    qpFactory->acceptIncomingConnection(feature_buffer, sizeof(infinity::memory::RegionToken));
                }
                // we will do better cleanup job laster
                while(1){
                    std::this_thread::sleep_for(std::chrono::seconds(1)); // 1s
                }
            }


    };
} // namespace qvf



void register_dist_tensor(pybind11::module &m) {
  
  //py::class_<qvf::DistTensor>(m, "QuiverDistTensor")
  //    .def(py::init<>());
  
}