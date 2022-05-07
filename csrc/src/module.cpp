
   

#include <torch/extension.h>

void register_dist_feature(pybind11::module &m);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  register_dist_feature(m);
}