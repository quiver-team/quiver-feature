

#include <qvf/com_endpoint.h>
#include <qvf/pipe.h>
#include <qvf/range.h>
#include <torch/extension.h>
void register_dist_tensor(pybind11::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  register_dist_tensor(m);
}
