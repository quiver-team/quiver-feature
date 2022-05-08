#include <qvf/dist_tensor.h>
#include <torch/extension.h>

void register_dist_tensor(pybind11::module& m) {
  // py::class_<qvf::DistTensor>(m, "QuiverDistTensor")
  //     .def(py::init<>());
}
