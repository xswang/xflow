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

#include "src/io/load_data_from_disk.h"
#include "src/base/thread_pool.h"
#include "src/base/base.h"
#include "ps/ps.h"

namespace xflow{
class MVMWorker{
 public:
  MVMWorker(const char *train_file,
           const char *test_file) :
           train_file_path(train_file),
           test_file_path(test_file) {
    kv_w = new ps::KVWorker<float>(0);
    kv_v = new ps::KVWorker<float>(1);
    base_ = new Base;
    core_num = std::thread::hardware_concurrency();
    pool_ = new ThreadPool(core_num);
  }
  ~MVMWorker() {}

  void calculate_pctr(int start, int end);
  void predict(ThreadPool* pool, int rank, int block);

  void calculate_gradient(std::vector<Base::sample_key>& all_keys,
                          std::vector<ps::Key>& unique_keys,
                          size_t start, size_t end,
                          std::vector<float>& v,
                          std::vector<float>& v_sum,
                          std::vector<float>& loss,
                          std::vector<float>& push_v_gradient);
  void calculate_loss(std::vector<float>& v,
                      std::vector<Base::sample_key>& all_keys,
                      std::vector<ps::Key>& unique_keys,
                      size_t start, size_t end,
                      std::vector<float>& v_sum,
                      std::vector<float>& loss);
  void update(int start, int end);
  void batch_training(ThreadPool* pool);
  void train();

 public:
  int epochs = 60;
 private:
  int rank;
  int core_num;
  int block_size = 2;

  std::atomic_llong num_batch_fly = {0};
  std::atomic_llong gradient_thread_finish_num = {0};
  std::atomic_llong calculate_pctr_thread_finish_num = {0};

  std::vector<Base::auc_key> auc_vec;
  std::vector<Base::auc_key> test_auc_vec;

  std::ofstream md;
  std::mutex mutex;
  Base* base_;
  ThreadPool* pool_;
  xflow::Data *train_data;
  xflow::Data *test_data;
  const char *train_file_path;
  const char *test_file_path;
  char train_data_path[1024];
  char test_data_path[1024];
  int v_dim_ = 10;
  ps::KVWorker<float>* kv_w;
  ps::KVWorker<float>* kv_v;
};//end class worker
}
