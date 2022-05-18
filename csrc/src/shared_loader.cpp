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

std::tuple<at::DataPtr, size_t> qvf::SharedLoader::getRecord(
    const std::string& name) {
  std::lock_guard<std::mutex> guard(reader.*RB(reader_lock_));
  size_t key = (reader.*RB(getRecordID))(name);
  mz_zip_archive_file_stat stat;
  mz_zip_reader_file_stat((reader.*RB(ar_)).get(), key, &stat);
  (reader.*RB(valid))("retrieving file meta-data for ", name.c_str());
  at::DataPtr retval = new_fd_storage(stat.m_uncomp_size);
  mz_zip_reader_extract_to_mem((reader.*RB(ar_)).get(), key, retval.get(),
                               stat.m_uncomp_size, 0);
  (reader.*RB(valid))("reading file ", name.c_str());

  return std::make_tuple(std::move(retval), stat.m_uncomp_size);
}
