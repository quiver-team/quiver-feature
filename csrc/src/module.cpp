

#include <infinity/infinity.h>
#include <qvf/qvf.h>
#include <torch/extension.h>

void register_TensorEndPoint(pybind11::module& m);
void register_DistTensorServer(pybind11::module& m);
void register_PipeParam(pybind11::module& m);
void register_DistTensorClient(pybind11::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  register_TensorEndPoint(m);
  register_DistTensorServer(m);
  register_PipeParam(m);
  register_DistTensorClient(m);
}
