#include <ATen/ATen.h>
#include <infinity/infinity.h>
#include <qvf/qvf.h>
#include <torch/extension.h>
#include <torch/jit.h>
#include <torch/serialize.h>
#include <torch/torch.h>

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
      .def(py::init<int, int, int, int>());
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

void register_SharedStorageReader(pybind11::module& m) {
  class BufferAdapter : public caffe2::serialize::ReadAdapterInterface {
   public:
    BufferAdapter(const py::object& buffer) : buffer_(buffer) {
      // Jump to the end of the buffer to get its size
      auto current = buffer.attr("tell")();
      start_offset_ = py::cast<size_t>(current);
      buffer.attr("seek")(current, py::module::import("os").attr("SEEK_END"));
      size_ = py::cast<size_t>(buffer.attr("tell")()) - start_offset_;
      buffer.attr("seek")(current);

      // If we can read directly into a buffer, do that instead of an extra copy
      use_readinto_ = py::hasattr(buffer, "readinto");
    }

    size_t size() const override { return size_; }

    THPObjectPtr getMemview(void* buf, size_t n) const {
      THPObjectPtr memview(PyMemoryView_FromMemory(reinterpret_cast<char*>(buf),
                                                   n, PyBUF_WRITE));
      if (!memview) {
        throw python_error();
      }
      return memview;
    }

    size_t read(uint64_t pos,
                void* buf,
                size_t n,
                const char* what) const override {
      // Seek to desired position (NB: this has to be a Py_ssize_t or Python
      // throws a weird error)
      Py_ssize_t absolute_pos = start_offset_ + pos;
      buffer_.attr("seek")(absolute_pos);

      if (use_readinto_) {
        auto memview = getMemview(buf, n);
        auto res =
            PyObject_CallMethod(buffer_.ptr(), "readinto", "O", memview.get());
        if (res) {
          int64_t i = static_cast<int64_t>(PyLong_AsLongLong(res));
          if (i > 0) {
            return i;
          }
        }
      }

      // Read bytes into `buf` from the buffer
      std::string bytes = py::cast<std::string>(buffer_.attr("read")(n));
      std::copy(bytes.data(), bytes.data() + bytes.size(),
                reinterpret_cast<char*>(buf));
      return bytes.size();
    }

    py::object buffer_;
    size_t size_;
    size_t start_offset_;
    bool use_readinto_;
  };
  py::class_<qvf::SharedLoader, std::shared_ptr<qvf::SharedLoader>>(
      m, "SharedTensorLoader")
      .def(py::init<std::string>())
      .def(py::init([](const py::object& buffer) {
        auto adapter = std::make_unique<BufferAdapter>(buffer);
        return std::make_shared<qvf::SharedLoader>(std::move(adapter));
      }))
      .def("get_record",
           [](qvf::SharedLoader& self, const std::string& key) {
             at::DataPtr data;
             size_t size = 0;
             std::tie(data, size) = self.getRecord(key);
             return py::bytes(reinterpret_cast<const char*>(data.get()), size);
           })
      .def("has_record",
           [](qvf::SharedLoader& self, const std::string& key) {
             return self.hasRecord(key);
           })
      .def("get_storage_from_record",
           [](qvf::SharedLoader& self, const std::string& key, size_t numel,
              py::object data_type_obj) {
             at::DataPtr data(std::get<0>(self.getRecord(key)));
             auto scalar_type =
                 reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;

             c10::Storage storage(c10::Storage::use_byte_size_t(),
                                  numel * elementSize(scalar_type),
                                  std::move(data),
                                  /*allocator=*/nullptr,
                                  /*resizable=*/false);
             auto ptr =
                 c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                     std::move(storage), at::DispatchKeySet(),
                     at::CPU(scalar_type).typeMeta());
             return at::Tensor(std::move(ptr));
           })
      .def("get_all_records",
           [](qvf::SharedLoader& self) { return self.getAllRecords(); });
}