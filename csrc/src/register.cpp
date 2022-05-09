#include <infinity/infinity.h>
#include <qvf/qvf.h>
#include <torch/extension.h>

void register_TensorEndPoint(pybind11::module& m) {
  // define TensorEndPoint
  py::class_<qvf::TensorEndPoint>(m, "TensorEndPoint")
      .def(py::init<int, std::string, int, int64_t, int64_t>());
}

void register_ComEndPoint(pybind11::module& m) {
  // define ComEndPoint
  py::class_<qvf::ComEndPoint>(m, "ComEndPoint")
      .def(py::init<int, std::string, int>());
}

void register_DistTensorServer(pybind11::module& m) {
  // define TensorEndPoint
  py::class_<qvf::DistTensorServer>(m, "DistTensorServer")
      .def(py::init<int, int, int>())
      .def("serve_tensor", &qvf::DistTensorServer::serve_tensor,
           py::call_guard<py::gil_scoped_release>())
      .def("join", &qvf::DistTensorServer::join,
           py::call_guard<py::gil_scoped_release>());
}

void register_PipeParam(pybind11::module& m) {
  py::class_<qvf::PipeParam>(m, "PipeParam")
      .def(py::init<int, int, int, int, int>());
}

void register_DistTensorClient(pybind11::module& m) {
  py::class_<qvf::DistTensorClient>(m, "DistTensorClient")
      .def(py::init<int, std::vector<qvf::ComEndPoint>, qvf::PipeParam>())
      .def("create_registered_float32_tensor",
           &qvf::DistTensorClient::create_registered_float32_tensor,
           py::call_guard<py::gil_scoped_release>())
      .def("register_float32_tensor",
           &qvf::DistTensorClient::register_float32_tensor,
           py::call_guard<py::gil_scoped_release>())
      .def("create_registered_float32_tensor_cuda",
           &qvf::DistTensorClient::create_registered_float32_tensor_cuda,
           py::call_guard<py::gil_scoped_release>())

      .def("sync_read", &qvf::DistTensorClient::sync_read,
           py::call_guard<py::gil_scoped_release>());
}
