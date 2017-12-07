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

#include "io/load_data_from_local.h"
#include "threadpool/thread_pool.h"
#include "ps.h"


class W{
 public:
  W(const char *train_file,
    const char *test_file) :
                           train_file_path(train_file),
                           test_file_path(test_file) {
    kv_ = new ps::KVWorker<float>(0);
  }
  ~W() {}

  float sigmoid(float x){
    if(x < -30) return 1e-6;
    else if(x > 30) return 1.0;
    else{
      double ex = pow(2.718281828, x);
      return ex / (1.0 + ex);
    }
  }
  struct sample_key{
    size_t fid;
    int sid;
  };
  static bool sort_finder(const sample_key& a, const sample_key& b){
    return a.fid < b.fid;
  }
  static bool unique_finder(const sample_key& a, const sample_key& b){
    return a.fid == b.fid;
  }
  void calculate_batch_gradient_threadpool(int start, int end){
    size_t idx = 0; float pctr = 0;
    auto all_keys = std::vector<sample_key>();
    auto unique_keys = std::vector<ps::Key>();;
    int line_num = 0;
    for(int row = start; row < end; ++row){
      int sample_size = train_data->fea_matrix[row].size();
      sample_key sk;
      sk.sid = line_num;
      for(int j = 0; j < sample_size; ++j){
        idx = train_data->fea_matrix[row][j].fid;
        sk.fid = idx;
        all_keys.push_back(sk);
        (unique_keys).push_back(idx);
      }
      ++line_num;
    }
    std::sort(all_keys.begin(), all_keys.end(), W::sort_finder);
    std::sort((unique_keys).begin(), (unique_keys).end());
    (unique_keys).erase(unique((unique_keys).begin(), (unique_keys).end()), (unique_keys).end());
    int keys_size = (unique_keys).size();

    auto w = std::vector<float>();
    kv_->Wait(kv_->Pull(unique_keys, &(w)));

    auto wx = std::vector<float>(end - start);
    for(int j = 0, i = 0; j < all_keys.size();){
      size_t allkeys_fid = all_keys[j].fid;
      size_t weight_fid = (unique_keys)[i];
      if(allkeys_fid == weight_fid){
        wx[all_keys[j].sid] += (w)[i];
        ++j;
      }
      else if(allkeys_fid > weight_fid){
        ++i;
      }
    }
    for(int i = 0; i < wx.size(); i++){
      pctr = sigmoid(wx[i]);
      float loss = pctr - train_data->label[start++];
      wx[i] = loss;
    }

    auto push_gradient = std::vector<float>(keys_size);
    for(int j = 0, i = 0; j < all_keys.size();){
      size_t allkeys_fid = all_keys[j].fid;
      size_t gradient_fid = (unique_keys)[i];
      int sid = all_keys[j].sid;
      if(allkeys_fid == gradient_fid){
        (push_gradient)[i] += wx[sid];
        ++j;
      }
      else if(allkeys_fid > gradient_fid){
        ++i;
      }
    }
    for(size_t i = 0; i < (push_gradient).size(); ++i){
      (push_gradient)[i] /= 1.0 * line_num;
    }

    kv_->Wait(kv_->Push(unique_keys, push_gradient));
    --gradient_thread_finish_num;
  }

  void batch_learning_threadpool(){
    ThreadPool pool(core_num);
    for(int epoch = 0; epoch < epochs; ++epoch){
      dml::LoadData train_data_loader(train_data_path, block_size<<20);
      train_data = &(train_data_loader.m_data);
      int block = 0;
      while(1){
        train_data_loader.load_minibatch_hash_data_fread();
        if(train_data->fea_matrix.size() <= 0) break;
        int thread_size = train_data->fea_matrix.size() / core_num;
        gradient_thread_finish_num = core_num;
        for(int i = 0; i < core_num; ++i){
          int start = i * thread_size;
          int end = (i + 1)* thread_size;
          pool.enqueue(std::bind(&W::calculate_batch_gradient_threadpool, this, start, end));
        }
        while(gradient_thread_finish_num > 0){
          usleep(5);
        }
        ++block;
      }
      std::cout << "epoch : " << epoch << std::endl;
      train_data = NULL;
    }
  }

  void P(){
    rank = ps::MyRank();
    std::cout << "my rank is = " << rank << std::endl;
    snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
    core_num = std::thread::hardware_concurrency();
    core_num = 1;
    batch_learning_threadpool();
    std::cout<<"train end......"<<std::endl;
  }

 public:
  int rank;
  int core_num;
  int batch_num;
  int block_size = 2;
  int epochs = 100;

  std::atomic_llong num_batch_fly = {0};
  std::atomic_llong gradient_thread_finish_num = {0};
  std::atomic_llong calculate_pctr_thread_finish_num = {0};

  float logloss = 0.0;
  float rmse = 0.0;

  std::ofstream md;
  std::mutex mutex;
  dml::Data *train_data;
  dml::Data *test_data;
  const char *train_file_path;
  const char *test_file_path;
  char train_data_path[1024];
  char test_data_path[1024];
  float bias = 0.0;
  ps::KVWorker<float>* kv_;
};//end class worker

