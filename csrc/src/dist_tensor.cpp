#include <qvf/pipe.h>
#include <qvf/com_endpoint.h>
#include <qvf/range.h>
#include <vector>
#include <torch/extension.h>
namespace qvf{
    class DistTensor{
        private:
            std::vector<Pipe> pipes;
            std::vector<Range> ranges;
            std::vector<ComEndPoint> com_endpoints;
        
        public:
            DistTensor(){}

    };
} // namespace qvf



void register_dist_tensor(pybind11::module &m) {
  
  //py::class_<qvf::DistTensor>(m, "QuiverDistTensor")
  //    .def(py::init<>());
  
}