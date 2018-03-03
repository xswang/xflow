/*
 * mvm_worker.cc
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <algorithm>
#include <ctime>
#include <iostream>

#include <mutex>
#include <functional>
#include <random>
#include <string>
#include <time.h>
#include <unistd.h>
#include <memory>
#include <immintrin.h>

#include "src/model/mvm/mvm_worker.h"

namespace xflow {
void MVMWorker::calculate_pctr(int start, int end) {
  auto all_keys = std::vector<Base::sample_key>();
  auto unique_keys = std::vector<ps::Key>();
  auto sample_fgid_num = std::vector<int>(end-start);
  int line_num = 0;
  for (int row = start; row < end; ++row) {
    int sample_size = test_data->fea_matrix[row].size();
    Base::sample_key sk;
    sk.sid = line_num;
    auto sample_fgid_map = std::map<size_t, int>();
    for (int j = 0; j < sample_size; ++j) {
      size_t fgid = test_data->fea_matrix[row][j].fgid;
      size_t idx = test_data->fea_matrix[row][j].fid;
      sk.fgid = fgid;
      sample_fgid_map.insert({fgid, 1});
      sk.fid = idx;
      all_keys.push_back(sk);
      (unique_keys).push_back(idx);
    }
    sample_fgid_num[line_num] = sample_fgid_map.size();
    ++line_num;
  }
  std::sort(all_keys.begin(), all_keys.end(), base_->sort_finder);
  std::sort((unique_keys).begin(), (unique_keys).end());
  (unique_keys).erase(unique((unique_keys).begin(), (unique_keys).end()),
                      (unique_keys).end());
  auto v = std::vector<float>();
  kv_v->Wait(kv_v->Pull(unique_keys, &v));

  auto v_sum = std::vector<std::vector<std::vector<float>>>();
  for (int k = 0; k < v_dim_; ++k) {
    auto k_sum = std::vector<std::vector<float>>();
    for (int i = 0; i < sample_fgid_num.size(); ++i) {
      auto fg_num = std::vector<float>(sample_fgid_num[i]);
      k_sum.push_back(fg_num);
    }
    v_sum.push_back(k_sum);
  }
  auto v_multi = std::vector<std::vector<float>>();
  for (int k = 0; k < v_dim_; ++k) {
    auto k_multi = std::vector<float>(end-start, 1.0);
    v_multi.push_back(k_multi);
  }
  for (size_t k = 0; k < v_dim_; ++k) {
    for (size_t j = 0, i = 0; j < all_keys.size();) {
      size_t allkeys_fid = all_keys[j].fid;
      size_t weight_fid = unique_keys[i];
      if (allkeys_fid == weight_fid) {
        size_t sid = all_keys[j].sid;
        size_t key_fgid = all_keys[j].fgid;
        key_fgid = 1;
        float v_weight = v[i * v_dim_ + k];
        v_sum[k][sid][key_fgid] += v_weight;
        ++j;
      } else if (allkeys_fid > weight_fid) {
        ++i;
      }
    }
  }
  for (int k = 0; k < v_dim_; ++k) {
    for (size_t i = 0; i < end - start; ++i) {
      for (size_t j = 0; j < v_sum[k][i].size(); ++j) {
        v_multi[k][i] *= (1.0 + v_sum[k][i][j]);
      }
    }
  }

  auto v_y = std::vector<float>(end-start);
  for (int i = 0; i < end-start; ++i) {
    for (int j = 0; j < v_multi.size(); ++j)
      v_y[i] += v_multi[j][i];
  }
  for (int i = 0; i < v_multi.size(); ++i) {
    float pctr = base_->sigmoid(v_y[i]);
    Base::auc_key ak;
    ak.label = test_data->label[start++];
    ak.pctr = pctr;
    mutex.lock();
    test_auc_vec.push_back(ak);
    md << pctr << "\t" << 1 - ak.label << "\t" << ak.label << std::endl;
    mutex.unlock();
  }
  --calculate_pctr_thread_finish_num;
}

void MVMWorker::predict(ThreadPool* pool, int rank, int block) {
  char buffer[1024];
  snprintf(buffer, 1024, "%d_%d", rank, block);
  std::string filename = buffer;
  md.open("pred_" + filename + ".txt");
  if (!md.is_open()) std::cout << "open pred file failure!" << std::endl;

  snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
  xflow::LoadData test_data_loader(test_data_path, ((size_t)4) << 20);
  test_data = &(test_data_loader.m_data);
  test_auc_vec.clear();
  while (true) {
    test_data_loader.load_minibatch_hash_data_fread();
    if (test_data->fea_matrix.size() <= 0) break;
    int thread_size = test_data->fea_matrix.size() / core_num;
    calculate_pctr_thread_finish_num = core_num;
    for (int i = 0; i < core_num; ++i) {
      int start = i * thread_size;
      int end = (i + 1)* thread_size;
      pool->enqueue(std::bind(&MVMWorker::calculate_pctr, this, start, end));
    }
    while (calculate_pctr_thread_finish_num > 0) usleep(10);
  }
  md.close();
  test_data = NULL;
  base_->calculate_auc(test_auc_vec);
}

void MVMWorker::calculate_gradient(std::vector<Base::sample_key>& all_keys,
    std::vector<ps::Key>& unique_keys,
    size_t start, size_t end,
    std::vector<float>& v,
    std::vector<std::vector<std::vector<float>>>& v_sum,
    std::vector<std::vector<float>>& v_multi,
    std::vector<float>& loss,
    std::vector<float>& push_v_gradient) {
  for (size_t k = 0; k < v_dim_; ++k) {
    for (int j = 0, i = 0; j < all_keys.size();) {
      size_t allkeys_fid = all_keys[j].fid;
      size_t weight_fid = unique_keys[i];
      int sid = all_keys[j].sid;
      int fgid = all_keys[j].fgid;
      if (allkeys_fid == weight_fid) {
        push_v_gradient[i * v_dim_ + k] +=
          loss[sid] *
          (1.0 * v_multi[k][sid] / (v_sum[k][sid][fgid]));
        ++j;
      } else if (allkeys_fid > weight_fid) {
        ++i;
      }
    }
  }

  size_t line_num = end - start;
  for (size_t i = 0; i < push_v_gradient.size(); ++i) {
    push_v_gradient[i] /= 1.0 * line_num;
    // std::cout << "push_v_gradient = " << push_v_gradient[i] << std::endl;
  }
}

void MVMWorker::calculate_loss(std::vector<float>& v,
    std::vector<Base::sample_key>& all_keys,
    std::vector<ps::Key>& unique_keys,
    size_t start, size_t end,
    std::vector<std::vector<std::vector<float>>>& v_sum,
    std::vector<std::vector<float>>& v_multi,
    std::vector<float>& loss) {
  float v_weight = 0.0;
  size_t sid = 0, key_fgid = 0;
  for (size_t k = 0; k < v_dim_; k++) {
    for (size_t j = 0, i = 0; j < all_keys.size();) {
      size_t allkeys_fid = all_keys[j].fid;
      size_t weight_fid = unique_keys[i];
      if (allkeys_fid == weight_fid) {
        sid = all_keys[j].sid;
        key_fgid = all_keys[j].fgid;
        v_weight = v[i * v_dim_ + k];
        v_sum[k][sid][key_fgid] += v_weight;
        ++j;
      } else if (allkeys_fid > weight_fid) {
        ++i;
      }
    }
  }
  for (size_t k = 0; k < v_dim_; ++k) {
    for (size_t i = 0; i < end - start; ++i) {
      for (size_t j = 0; j < v_sum[k][i].size(); ++j) {
        v_multi[k][i] *= (1.0 + v_sum[k][i][j]);
      }
    }
  }

  auto v_y = std::vector<float>(end-start);
  for (size_t i = 0; i < end - start; ++i) {
    for (size_t j = 0; j < v_multi.size(); ++j) {
      v_y[i] += v_multi[j][i];
    }
  }
  std::cout << "line 200 v_y[0] == " << v_y[0] << std::endl;

  for (int i = 0; i < end - start; i++) {
    float pctr = base_->sigmoid(v_y[i]);
    loss[i] = pctr - train_data->label[start++];
  }
}

void MVMWorker::update(int start, int end) {
  size_t idx = 0, fgid = 0;
  auto all_keys = std::vector<Base::sample_key>();
  auto unique_keys = std::vector<ps::Key>();
  auto sample_fgid_num = std::vector<int>(end-start);
  int line_num = 0;
  for (int row = start; row < end; ++row) {
    int sample_size = train_data->fea_matrix[row].size();
    Base::sample_key sk;
    sk.sid = line_num;
    auto sample_fgid_map = std::map<size_t, int>();
    for (int j = 0; j < sample_size; ++j) {
      fgid = train_data->fea_matrix[row][j].fgid;
      idx = train_data->fea_matrix[row][j].fid;
      sk.fgid = fgid;
      sample_fgid_map.insert({fgid, 1});
      sk.fid = idx;
      all_keys.push_back(sk);
      unique_keys.push_back(idx);
    }
    sample_fgid_num[line_num] = sample_fgid_map.size();
    ++line_num;
  }
  std::sort(all_keys.begin(), all_keys.end(), base_->sort_finder);
  std::sort((unique_keys).begin(), (unique_keys).end());
  (unique_keys).erase(unique((unique_keys).begin(), (unique_keys).end()),
                      unique_keys.end());
  int keys_size = unique_keys.size();

  auto v = std::vector<float>();
  kv_v->Wait(kv_v->Pull(unique_keys, &v));

  auto push_v_gradient = std::vector<float>(keys_size * v_dim_);

  auto loss = std::vector<float>(end - start);
  auto v_multi = std::vector<std::vector<float>>();
  for (size_t k = 0; k < v_dim_; ++k) {
    auto k_multi = std::vector<float>(end-start, 1.0);
    v_multi.push_back(k_multi);
  }
  auto v_sum = std::vector<std::vector<std::vector<float>>> ();
  for (int k = 0; k < v_dim_; ++k) {
    auto k_sum = std::vector<std::vector<float>>();
    for (int i = 0; i < sample_fgid_num.size(); ++i) {
      auto fg_num = std::vector<float>(sample_fgid_num[i]);
      k_sum.push_back(fg_num);
    }
    v_sum.push_back(k_sum);
  }
  calculate_loss(v, all_keys, unique_keys, start, end, v_sum, v_multi, loss);
  calculate_gradient(all_keys, unique_keys, start, end, v, v_sum, v_multi, loss,
                     push_v_gradient);

  kv_v->Wait(kv_v->Push(unique_keys, push_v_gradient));

  --gradient_thread_finish_num;
}

void MVMWorker::batch_training(ThreadPool* pool) {
  std::vector<ps::Key> key(1);
  std::vector<float> val_v(v_dim_);
  kv_v->Wait(kv_v->Push(key, val_v));
  for (int epoch = 0; epoch < epochs; ++epoch) {
    xflow::LoadData train_data_loader(train_data_path, block_size << 20);
    train_data = &(train_data_loader.m_data);
    int block = 0;
    while (1) {
      train_data_loader.load_minibatch_hash_data_fread();
      if (train_data->fea_matrix.size() <= 0) break;
      int thread_size = train_data->fea_matrix.size() / core_num;
      gradient_thread_finish_num = core_num;
      for (int i = 0; i < core_num; ++i) {
        int start = i * thread_size;
        int end = (i + 1)* thread_size;
        pool->enqueue(std::bind(&MVMWorker::update, this, start, end));
      }
      while (gradient_thread_finish_num > 0) {
        usleep(5);
      }
      ++block;
    }
    if ((epoch + 1) % 30 == 0) std::cout << "epoch : " << epoch << std::endl;
    train_data = NULL;
  }
}

void MVMWorker::train() {
  rank = ps::MyRank();
  std::cout << "my rank is = " << rank << std::endl;
  snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
  batch_training(pool_);
  if (rank == 0) {
    std::cout << "FM AUC: " << std::endl;
    predict(pool_, rank, 0);
  }
  std::cout << "train end......" << std::endl;
}
}  // namespace xflow
