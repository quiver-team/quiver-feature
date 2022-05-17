#include <ATen/MapAllocator.h>
#include <c10/core/CPUAllocator.h>
#include <qvf/shared_loader.h>
extern "C" {
#include <miniz/miniz.h>
}

#define RB(x) get(PyTorchStreamReader_##x())

at::DataPtr new_fd_storage(ptrdiff_t size) {
  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_EXCLUSIVE |
              at::ALLOCATOR_MAPPED_KEEPFD | at::ALLOCATOR_MAPPED_UNLINK;
  std::string handle = at::NewProcessWideShmHandle();
  auto sptr = at::MapAllocator::makeDataPtr(handle.c_str(), flags,
                                            size * sizeof(uint8_t), nullptr);

  return sptr;
}

size_t qvf::SharedLoader::getRecordID(const std::string& name) {
  std::string ss = reader.*RB(archive_name_plus_slash_) + name;
  size_t result = mz_zip_reader_locate_file((reader.*RB(ar_)).get(), ss.c_str(),
                                            nullptr, 0);
  //  valid("locating file ", name.c_str());
  return result;
}

std::tuple<at::DataPtr, size_t> qvf::SharedLoader::getRecord(
    const std::string& name) {
  std::lock_guard<std::mutex> guard(reader.*RB(reader_lock_));
//  std::cout<<"!!Get record of "<< name<<std::endl;
  size_t key = getRecordID(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat((reader.*RB(ar_)).get(), key, &stat);
  valid("retrieving file meta-data for ", name.c_str());
  at::DataPtr retval = new_fd_storage(stat.m_uncomp_size);
//    at::DataPtr retval = c10::GetCPUAllocator()->allocate(stat.m_uncomp_size);
//  std::cout << "m_uncomp_size is " << stat.m_uncomp_size << std::endl
//            << "shared_ptr is " << retval.get() << std::endl;
  mz_zip_reader_extract_to_mem((reader.*RB(ar_)).get(), key, retval.get(),
                               stat.m_uncomp_size, 0);
  valid("reading file ", name.c_str());

  return std::make_tuple(std::move(retval), stat.m_uncomp_size);
}

void qvf::SharedLoader::valid(const char* what, const char* info) {
  const auto err = mz_zip_get_last_error((reader.*RB(ar_)).get());
  TORCH_CHECK(err == MZ_ZIP_NO_ERROR, "PytorchStreamReader failed ", what, info,
              ": ", mz_zip_get_error_string(err));
}