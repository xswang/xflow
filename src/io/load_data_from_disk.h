/*
 * base.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SRC_IO_LOAD_DATA_FROM_DISK_H_
#define SRC_IO_LOAD_DATA_FROM_DISK_H_

#include <string.h>

#include <vector>
#include <set>

#include "src/io/io.h"

namespace xflow {
class LoadData : public IO{
 public:
  LoadData(const char *file_path, size_t block_size)
    : IO(file_path), buf(block_size) {}
  ~LoadData() {}

  void load_all_data();
  void load_all_hash_data();
  void load_minibatch_data(int num);
  void load_mibibatch_hash_data(int num);
  void load_minibatch_hash_data_fread();
 private:
  std::vector<char> buf;
 public:
  Data m_data;
};

}  // namespace xflow
#endif  // SRC_IO_LOAD_DATA_FROM_DISK_H_
