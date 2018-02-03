/*
 * lr_worker.cc
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include <time.h>
#include <unistd.h>
#include <immintrin.h>

#include <algorithm>
#include <ctime>
#include <iostream>

#include <mutex>
#include <functional>
#include <random>
#include <string>
#include <memory>

#include "src/model/lr/lr_worker.h"

namespace xflow {
void LRWorker::calculate_pctr(int start, int end) {
    auto all_keys = std::vector<Base::sample_key>();
    auto unique_keys = std::vector<ps::Key>();
    int line_num = 0;
    for (int row = start; row < end; ++row) {
      int sample_size = test_data->fea_matrix[row].size();
      Base::sample_key sk;
      sk.sid = line_num;
      for (int j = 0; j < sample_size; ++j) {
        size_t idx = test_data->fea_matrix[row][j].fid;
        sk.fid = idx;
        all_keys.push_back(sk);
        unique_keys.push_back(idx);
      }
      ++line_num;
    }
    std::sort(all_keys.begin(), all_keys.end(), base_->sort_finder);
    std::sort(unique_keys.begin(), unique_keys.end());
    (unique_keys).erase(unique(unique_keys.begin(), unique_keys.end()),
                        unique_keys.end());
    int keys_size = (unique_keys).size();
    auto w = std::vector<float>(keys_size);
    kv_w_->Wait(kv_w_->Pull(unique_keys, &(w)));
    auto wx = std::vector<float>(line_num);
    for (int j = 0, i = 0; j < all_keys.size(); ) {
      size_t allkeys_fid = all_keys[j].fid;
      size_t weight_fid = (unique_keys)[i];
      if (allkeys_fid == weight_fid) {
        wx[all_keys[j].sid] += (w)[i];
        ++j;
      }
      else if (allkeys_fid > weight_fid) {
        ++i;
      }
    }
    for (int i = 0; i < wx.size(); ++i) {
      float pctr = base_->sigmoid(wx[i]);
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

void LRWorker::predict(ThreadPool* pool, int rank, int block) {
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
        pool->enqueue(std::bind(&LRWorker::calculate_pctr, this, start, end));
      }
      while (calculate_pctr_thread_finish_num > 0) usleep(10);
    }
    md.close();
    test_data = NULL;
    base_->calculate_auc(test_auc_vec);
  }

void LRWorker::calculate_gradient(std::vector<Base::sample_key>& all_keys,
                          std::vector<ps::Key>& unique_keys,
                          std::vector<float>& loss,
                          std::vector<float>& push_gradient) {
    for (int j = 0, i = 0; j < all_keys.size(); ) {
      size_t allkeys_fid = all_keys[j].fid;
      size_t gradient_fid = (unique_keys)[i];
      int sid = all_keys[j].sid;
      if (allkeys_fid == gradient_fid) {
        push_gradient[i] += loss[sid];
        ++j;
      }
      else if (allkeys_fid > gradient_fid) {
        ++i;
      }
    }
    for (size_t i = 0; i < (push_gradient).size(); ++i) {
      push_gradient[i] /= 1.0 * loss.size();
    }
  }

void LRWorker::calculate_loss(std::vector<float>& w,
                      std::vector<Base::sample_key>& all_keys,
                      std::vector<ps::Key>& unique_keys,
                      size_t start,
                      size_t end,
                      std::vector<float>& loss) {
    auto wx = std::vector<float>(end - start);
    for (int j = 0, i = 0; j < all_keys.size(); ) {
      size_t allkeys_fid = all_keys[j].fid;
      size_t weight_fid = (unique_keys)[i];
      if (allkeys_fid == weight_fid) {
        wx[all_keys[j].sid] += (w)[i];
        ++j;
      }
      else if (allkeys_fid > weight_fid) {
        ++i;
      }
    }
    for (int i = 0; i < wx.size(); i++) {
      float pctr = base_->sigmoid(wx[i]);
      loss[i] = pctr - train_data->label[start++];
    }
  }

void LRWorker::update(int start, int end) {
    size_t idx = 0;
    auto all_keys = std::vector<Base::sample_key>();
    auto unique_keys = std::vector<ps::Key>();;
    int line_num = 0;
    for (int row = start; row < end; ++row) {
      int sample_size = train_data->fea_matrix[row].size();
      Base::sample_key sk;
      sk.sid = line_num;
      for (int j = 0; j < sample_size; ++j) {
        idx = train_data->fea_matrix[row][j].fid;
        sk.fid = idx;
        all_keys.push_back(sk);
        unique_keys.push_back(idx);
      }
      ++line_num;
    }
    std::sort(all_keys.begin(), all_keys.end(), base_->sort_finder);
    std::sort(unique_keys.begin(), unique_keys.end());
    unique_keys.erase(unique(unique_keys.begin(), unique_keys.end()),
                        unique_keys.end());
    int keys_size = (unique_keys).size();

    auto w = std::vector<float>(keys_size);
    auto push_gradient = std::vector<float>(keys_size);
    kv_w_->Wait(kv_w_->Pull(unique_keys, &(w)));
    auto loss = std::vector<float>(end - start);
    calculate_loss(w, all_keys, unique_keys, start, end, loss);
    calculate_gradient(all_keys, unique_keys, loss, push_gradient);

    kv_w_->Wait(kv_w_->Push(unique_keys, push_gradient));
    --gradient_thread_finish_num;
  }

void LRWorker::batch_training(ThreadPool* pool) {
    std::vector<ps::Key> keys(1);
    std::vector<float> vals(1);
    kv_w_->Wait(kv_w_->Push(keys, vals));
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
          pool->enqueue(std::bind(&LRWorker::update, this, start, end));
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

void LRWorker::train() {
    rank = ps::MyRank();
    std::cout << "my rank is = " << rank << std::endl;
    snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
    batch_training(pool_);
    if (rank == 0) {
      std::cout << "LR AUC: " << std::endl;
      predict(pool_, rank, 0);
    }
    std::cout << "train end......" << std::endl;
  }
}  // namespace xflow
