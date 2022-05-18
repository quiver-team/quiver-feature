//
// Created by joker on 2022/5/15.
//

#ifndef QUIVER_FEATURE_SHAREDLOADER_H
#define QUIVER_FEATURE_SHAREDLOADER_H

#include <serialize/inline_container.h>
#include <serialize/read_adapter_interface.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace qvf {

using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;

template <typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type get(Tag) { return M; }
};

#define ROB_FIELD_FROM_READER(FieldType, FieldName)    \
  struct PyTorchStreamReader_##FieldName {             \
    typedef FieldType PyTorchStreamReader::*type;      \
    friend type get(PyTorchStreamReader_##FieldName);  \
  };                                                   \
  template struct Rob<PyTorchStreamReader_##FieldName, \
                      &PyTorchStreamReader::FieldName>

ROB_FIELD_FROM_READER(std::string, archive_name_plus_slash_);
ROB_FIELD_FROM_READER(std::unique_ptr<mz_zip_archive>, ar_);
ROB_FIELD_FROM_READER(std::mutex, reader_lock_);

#define ROB_MEMBER_FUNCTION(RetVal, FieldName, Args)   \
  struct PyTorchStreamReader_##FieldName {             \
    typedef RetVal(PyTorchStreamReader::*type) Args;   \
    friend type get(PyTorchStreamReader_##FieldName);  \
  };                                                   \
  template struct Rob<PyTorchStreamReader_##FieldName, \
                      &PyTorchStreamReader::FieldName>

ROB_MEMBER_FUNCTION(void, valid, (const char*, const char*));
ROB_MEMBER_FUNCTION(size_t, getRecordID, (const std::string& name));

struct TORCH_API SharedLoader {
  PyTorchStreamReader reader;
  explicit SharedLoader(const std::string& file_name) : reader(file_name) {}
  explicit SharedLoader(std::istream* in) : reader(in) {}
  explicit SharedLoader(std::shared_ptr<ReadAdapterInterface> in)
      : reader(in) {}
  std::tuple<at::DataPtr, size_t> getRecord(const std::string& name);
  size_t getRecordOffset(const std::string& name) {
    return reader.getRecordOffset(name);
  }
  bool hasRecord(const std::string& name) { return reader.hasRecord(name); }
  std::vector<std::string> getAllRecords() { return reader.getAllRecords(); }
};

}  // namespace qvf
#endif  // QUIVER_FEATURE_SHAREDLOADER_H
