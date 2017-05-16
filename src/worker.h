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

//#include "io/load_data_from_kafka.h"
#include "io/load_data_from_local.h"
#include "threadpool/thread_pool.h"
#include "ps.h"

#include <netdb.h>  
#include <net/if.h>  
#include <arpa/inet.h>  
#include <sys/ioctl.h>  
#include <sys/types.h>  
#include <sys/socket.h>  

namespace dmlc{
class Worker : public ps::App{
  public:
    Worker(const char *train_file, const char *test_file) : 
      train_file_path(train_file), test_file_path(test_file), isStartToRun(false){
    }
    ~Worker(){
      //delete train_data;
    } 
 
    virtual void ProcessRequest(ps::Message* request){
      // When scheduler notice worker: servers have finished saving model.
      if (request->task.msg()=="StartRun"){
        isStartToRun = true;
      }
    }

    float sigmoid(float x){
      if(x < -30) return 1e-6;
      else if(x > 30) return 1.0;
      else{
        double ex = pow(2.718281828, x);
        return ex / (1.0 + ex);
      }
    }

	virtual bool Run(){
      while (!isStartToRun){
        std::cout<<"Worker waiting for start to run"<<std::endl;
        sleep(1);
      }
      std::cout<<"Worker start to run"<<std::endl;
	  Process();
	}

    struct auc_key{
      int label;
      float pctr;
    };

    timespec time_diff(timespec start, timespec end){
      timespec tmp;
      tmp.tv_sec =  end.tv_sec - start.tv_sec;
      tmp.tv_nsec = end.tv_nsec - start.tv_nsec;
      return tmp;
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

    void calculate_auc(std::vector<auc_key>& auc_vec){
      std::sort(auc_vec.begin(), auc_vec.end(), [](const auc_key& a, const auc_key& b){
        return a.pctr > b.pctr;
      });
      float area = 0.0; 
	  int tp_n = 0;
      for(size_t i = 0; i < auc_vec.size(); ++i){
	    if(i % 500000 == 0) std::cout<<"auc_label = "<<auc_vec[i].label<<std::endl;
	    //if(i % 500000 == 0) std::cout<<"auc_label = "<<auc_vec[i].label<<" auc_pctr = "<<auc_vec[i].pctr<<std::endl;
        if(auc_vec[i].label == 1) tp_n += 1;
        else area += tp_n;
      }
      if (tp_n == 0 || tp_n == auc_vec.size()) std::cout<<"tp_n = "<<tp_n<<std::endl;
      else{
        area /= 1.0 * (tp_n * (auc_vec.size() - tp_n));
        std::cout<<"auc = "<<area<<"\ttp = "<<tp_n<<" fp = "<<(auc_vec.size() - tp_n)<<std::endl;
      }
    }

	void calculate_pctr(int start, int end){
      auto all_keys = std::vector<sample_key>();
      auto unique_keys = std::make_shared<std::vector<ps::Key>>();
      int line_num = 0;
      for(int row = start; row < end; ++row) {
        int sample_size = test_data->fea_matrix[row].size();
        sample_key sk;
        sk.sid = line_num;
        for(int j = 0; j < sample_size; ++j) {
		  size_t idx = test_data->fea_matrix[row][j].fid;
          sk.fid = idx;
          all_keys.push_back(sk);
          (*unique_keys).push_back(idx);
		}
		++line_num;
      }
      std::sort(all_keys.begin(), all_keys.end(), Worker::sort_finder);
	  std::sort((*unique_keys).begin(), (*unique_keys).end());
	  (*unique_keys).erase(unique((*unique_keys).begin(), (*unique_keys).end()), (*unique_keys).end());
	  auto w = std::make_shared<std::vector<float>>();
	  int keys_size = (*unique_keys).size();
	  kv_.Wait(kv_.ZPull(unique_keys, &(*w)));
	  auto wx = std::vector<float>(line_num);

      for(int j = 0, i = 0; j < all_keys.size();){
        size_t allkeys_fid = all_keys[j].fid;
        size_t weight_fid = (*unique_keys)[i];
        if(allkeys_fid == weight_fid){
          wx[all_keys[j].sid] += (*w)[i];
          ++j;
        }
        else if(allkeys_fid > weight_fid){
          ++i;
        }
      }
      for(int i = 0; i < wx.size(); ++i){
        float pctr = sigmoid(wx[i]);
        auc_key ak;
        ak.label = test_data->label[start++];
        ak.pctr = pctr;
        mutex.lock();
        test_auc_vec.push_back(ak);
        md<<pctr<<"\t"<<ak.label<<std::endl;
		mutex.unlock();
      }
	  --calculate_pctr_thread_finish_num;
	}//calculate_pctr

    void predict(ThreadPool &pool, int rank, int block){
      char buffer[1024];
      snprintf(buffer, 1024, "%d_%d", rank, block);
      std::string filename = buffer;
      md.open("pred_" + filename + ".txt");
      if(!md.is_open()) std::cout<<"open pred file failure!"<<std::endl;

      snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
      dml::LoadData test_data_loader(test_data_path, ((size_t)4)<<30);
      test_data = &(test_data_loader.m_data);
      std::cout<<"alloc 4GB memory sucess!"<<std::endl;
      test_auc_vec.clear();
      while(true){
        test_data_loader.load_minibatch_hash_data_fread();
        std::cout<<"test_data size = "<<test_data->fea_matrix.size()<<std::endl;
        if(test_data->fea_matrix.size() <= 0) break;
        int thread_size = test_data->fea_matrix.size() / core_num;
        calculate_pctr_thread_finish_num = core_num;
        for(int i = 0; i < core_num; ++i){
          int start = i * thread_size;
          int end = (i + 1)* thread_size;
          pool.enqueue(std::bind(&Worker::calculate_pctr, this, start, end));
        }//end all batch
        while(calculate_pctr_thread_finish_num > 0) usleep(10);
        std::cout<<"test auc vec size in while = "<<test_auc_vec.size()<<std::endl;
      }//end while
      md.close();
      test_data = NULL;
      std::cout<<"test auc vec size out while = "<<test_auc_vec.size()<<std::endl;
      std::cout<<"block="<<block<<" ";
      calculate_auc(test_auc_vec);
    }//end predict 


  void predict_kafka(ThreadPool &pool, int rank){
      char buffer[1024];
      static int predict_id = 0;
      ++predict_id;
      sprintf(buffer, "%d_%d", rank, predict_id);
      std::string filename = buffer;
      md.open("pred_" + filename + ".txt");
      if(!md.is_open()) std::cout<<"open pred file failure!"<<std::endl;
      std::cout << "predict file open: " << filename << std::endl;

      test_data = train_data; // For kafka, just use train data as test data.
      test_auc_vec.clear();
      {
        std::cout<<"predict_kafka: test_data size = "<<test_data->fea_matrix.size()<<std::endl;
        if(test_data->fea_matrix.size() <= 0) return;
        int thread_size = test_data->fea_matrix.size() / core_num;
        calculate_pctr_thread_finish_num = core_num;
        for(int i = 0; i < core_num; ++i){
          int start = i * thread_size;
          int end = (i + 1)* thread_size;
          pool.enqueue(std::bind(&Worker::calculate_pctr, this, start, end));
        }//end all batch
        while(calculate_pctr_thread_finish_num > 0) usleep(10);
        std::cout<<"test auc vec size in while = "<<test_auc_vec.size()<<std::endl;
      }
      std::cout<<"test auc vec size out while = "<<test_auc_vec.size()<<std::endl;
      calculate_auc(test_auc_vec);

      md.close();
    }//end predict 


    void zpush_callback(std::vector<sample_key> all_keys, std::shared_ptr<std::vector<ps::Key>> unique_keys, std::shared_ptr<std::vector<float>> w, int start, int end){
      auto wx = std::vector<float>(end - start + 1);
      for(int j = 0, i = 0; j < all_keys.size();){
        size_t allkeys_fid = all_keys[j].fid;
        size_t weight_fid = (*unique_keys)[i];
        if(allkeys_fid == weight_fid){
          wx[all_keys[j].sid] += (*w)[i];
          ++j;
        }
        else if(allkeys_fid > weight_fid){
          ++i;
        }
      }

      for(int i = 0; i < wx.size(); i++){
        float pctr = sigmoid(wx[i]);
        float loss = pctr - train_data->label[start++];
        wx[i] = loss;
      }

      auto push_gradient = std::make_shared<std::vector<float> > ((*unique_keys).size());
      for(int j = 0, i = 0; j < all_keys.size();){
        size_t allkeys_fid = all_keys[j].fid;
        size_t gradient_fid = (*unique_keys)[i];
        int sid = all_keys[j].sid;
        if(allkeys_fid == gradient_fid){
          (*push_gradient)[i] += wx[sid];
          ++j;
        }
        else if(allkeys_fid > gradient_fid){
          ++i;
        }
      }
      ps::SyncOpts callback_push;
      callback_push.callback = [this](){
        --num_batch_fly;
      };
      kv_.ZPush(unique_keys, push_gradient, callback_push);
    }    

    void calculate_batch_gradient_callback(ThreadPool &pool, int start, int end){
      size_t idx = 0; float pctr = 0;
      auto all_keys = std::vector<sample_key>();
      auto unique_keys = std::make_shared<std::vector<ps::Key>> ();;
      int line_num = 0;
      for(int row = start; row < end; ++row){
        int sample_size = train_data->fea_matrix[row].size();
        sample_key sk;
        sk.sid = line_num;
        for(int j = 0; j < sample_size; ++j){//for one instance
          idx = train_data->fea_matrix[row][j].fid;
          sk.fid = idx;
          all_keys.push_back(sk);
          (*unique_keys).push_back(idx);
        }
        ++line_num;
      }
      std::sort(all_keys.begin(), all_keys.end(), Worker::sort_finder);
      std::sort((*unique_keys).begin(), (*unique_keys).end());
      (*unique_keys).erase(unique((*unique_keys).begin(), (*unique_keys).end()), (*unique_keys).end());

      auto w = std::make_shared<std::vector<float>>();

      ps::SyncOpts pull_callback;
      pull_callback.callback = [this, &pool, all_keys, unique_keys, w, start, end](){
        pool.enqueue(std::bind(&Worker::zpush_callback, this, all_keys, unique_keys, w, start, end));
      };
      kv_.ZPull(unique_keys, &(*w), pull_callback);
    }//calculate_batch_gradient_callback

    void batch_learning_callback(){
      dml::LoadData train_data_loader(train_data_path, 1);
      train_data = &(train_data_loader.m_data);
      train_data_loader.load_all_data();
      ThreadPool pool(core_num);

      batch_num = train_data->fea_matrix.size() / block_size;
      std::cout<<"batch_num : "<<batch_num<<std::endl;
      timespec allstart, allend, allelapsed;
      for(int epoch = 0; epoch < epochs; ++epoch){
        clock_gettime(CLOCK_MONOTONIC, &allstart);
        send_key_numbers = 0;
        for(int i = 0; i < batch_num; ++i){
          int all_start = i * block_size;
          int all_end = (i + 1)* block_size;
          int thread_batch = block_size / core_num;
          for(int j = 0; j < core_num; ++j){
            int start = all_start + j * thread_batch;
            int end = all_start + (j + 1) * thread_batch;
            calculate_batch_gradient_callback(pool, start, end);
            //pool.enqueue(std::bind(&Worker::calculate_batch_gradient_callback, this, pool, start, end));
            //if(i == 0) usleep(4000);
          }
        }//end all batch
        clock_gettime(CLOCK_MONOTONIC, &allend);
        allelapsed = time_diff(allstart, allend);
        std::cout<<"rank "<<rank<<" per process : "<<train_data->fea_matrix.size() * 1e9 * 1.0 / (allelapsed.tv_sec * 1e9 + allelapsed.tv_nsec)<<std::endl;
        std::cout<<"rank "<<rank<<" send_key_number avage: "<<send_key_numbers * 1.0 / (batch_num * core_num)<<std::endl;
      }//end all epoch
      train_data = NULL;
    }//end batch_learning_callback

    void calculate_batch_gradient_threadpool(int start, int end){
      timespec all_start, all_end, all_elapsed_time;
      clock_gettime(CLOCK_MONOTONIC, &all_start);
      size_t idx = 0; float pctr = 0;
      auto all_keys = std::vector<sample_key>();
      auto unique_keys = std::make_shared<std::vector<ps::Key>> ();;
      int line_num = 0;
      for(int row = start; row < end; ++row){
        int sample_size = train_data->fea_matrix[row].size();
        sample_key sk;
        sk.sid = line_num;
        for(int j = 0; j < sample_size; ++j){//for one instance
          idx = train_data->fea_matrix[row][j].fid;
          sk.fid = idx;
          all_keys.push_back(sk);
          (*unique_keys).push_back(idx);
        }
        ++line_num;
      }
      std::sort(all_keys.begin(), all_keys.end(), Worker::sort_finder);
      std::sort((*unique_keys).begin(), (*unique_keys).end());
      (*unique_keys).erase(unique((*unique_keys).begin(), (*unique_keys).end()), (*unique_keys).end());
      int keys_size = (*unique_keys).size();

      auto w = std::make_shared<std::vector<float>>();
      timespec pull_start_time, pull_end_time, pull_elapsed_time;
      clock_gettime(CLOCK_MONOTONIC, &pull_start_time);
      kv_.Wait(kv_.ZPull(unique_keys, &(*w)));
      clock_gettime(CLOCK_MONOTONIC, &pull_end_time);
      pull_elapsed_time = time_diff(pull_start_time, pull_end_time);

      auto wx = std::vector<float>(end - start);
      for(int j = 0, i = 0; j < all_keys.size();){
        size_t allkeys_fid = all_keys[j].fid;
        size_t weight_fid = (*unique_keys)[i];
        if(allkeys_fid == weight_fid){
          wx[all_keys[j].sid] += (*w)[i];
          ++j;
        }
        else if(allkeys_fid > weight_fid){ 
          ++i;
        }
      }//end for
      for(int i = 0; i < wx.size(); i++){
        pctr = sigmoid(wx[i]);
        float loss = pctr - train_data->label[start++];
        wx[i] = loss;
      }

      auto push_gradient = std::make_shared<std::vector<float> > (keys_size);
      for(int j = 0, i = 0; j < all_keys.size();){
        size_t allkeys_fid = all_keys[j].fid;
        size_t gradient_fid = (*unique_keys)[i];
        int sid = all_keys[j].sid;
        if(allkeys_fid == gradient_fid){
          (*push_gradient)[i] += wx[sid];
          ++j;
        }
        else if(allkeys_fid > gradient_fid){
          ++i;
        }
      }
      for(size_t i = 0; i < (*push_gradient).size(); ++i){
        (*push_gradient)[i] /= 1.0 * line_num;
      }

      timespec push_start_time, push_end_time, push_elapsed_time;
      clock_gettime(CLOCK_MONOTONIC, &push_start_time);
      kv_.Wait(kv_.ZPush(unique_keys, push_gradient));//put gradient to servers;
      clock_gettime(CLOCK_MONOTONIC, &push_end_time);
      push_elapsed_time = time_diff(push_start_time, push_end_time);
      clock_gettime(CLOCK_MONOTONIC, &all_end);
      all_elapsed_time = time_diff(all_start, all_end);

      all_time += all_elapsed_time.tv_sec * 1e9 + all_elapsed_time.tv_nsec;
      all_pull_time += pull_elapsed_time.tv_sec * 1e9 + pull_elapsed_time.tv_nsec;
      all_push_time += push_elapsed_time.tv_sec * 1e9 + push_elapsed_time.tv_nsec;
      send_key_numbers += keys_size;
      --calculate_batch_gradient_thread_finish_num;
    }

    void batch_learning_threadpool(){ // Load data from local disk file. For offline benchmark test.
      ThreadPool pool(core_num);
      timespec allstart, allend, allelapsed;
      int train_count = 0;
      for(int epoch = 0; epoch < epochs; ++epoch){
        dml::LoadData train_data_loader(train_data_path, block_size<<20);
        train_data = &(train_data_loader.m_data);
        int block = 0;
        while(1){
          train_data_loader.load_minibatch_hash_data_fread(); // Load a minibatch data to buffer.
          if(train_data->fea_matrix.size() <= 0) break; // No data read, then stop.
          int thread_size = train_data->fea_matrix.size() / core_num; // Partition the minibatch to multi-threads.
          calculate_batch_gradient_thread_finish_num = core_num;
          for(int i = 0; i < core_num; ++i){
            int start = i * thread_size;
            int end = (i + 1)* thread_size;
            pool.enqueue(std::bind(&Worker::calculate_batch_gradient_threadpool, this, start, end));
          }//end all batch
          while(calculate_batch_gradient_thread_finish_num > 0){ // Wait for all training threads to finish.
            usleep(10);
          }
          train_count += train_data->fea_matrix.size();
          if((rank == 0) && ((block + 1) % 3 == 0)) 
          {
            std::cout << "Trainied count = " << train_count << std::endl;
            train_count = 0;
            predict(pool, rank, block);
          }
          ++block;
        }//end mini-batch
        train_data = NULL;
      }//end epoch
    }//end batch_learning_threadpool

    /*
    void online_learning_threadpool(){ // Load data from kafka. Never stop.
       ThreadPool pool(core_num);

       dml::LoadData_from_kafka train_data_kafka;
       train_data = &(train_data_kafka.m_data);
       while(1){
         train_data_kafka.load_data_from_kafka();
         int thread_size = train_data->fea_matrix.size() / core_num;
         calculate_batch_gradient_thread_finish_num = core_num;
         for(int i = 0; i < core_num; ++i){
           int start = i * thread_size;
           int end = (i + 1)* thread_size;
std::cout << "Start thread (" << start << ", " << end << ")" << std::endl;
           pool.enqueue(std::bind(&Worker::calculate_batch_gradient_threadpool, this, start, end));
         }//end all batch
         while(calculate_batch_gradient_thread_finish_num > 0){
           usleep(10);
         }
std::cout << "All threads finished" << std::endl;
         predict_kafka(pool, rank);
       }//end while
       train_data = NULL;
    }
    */

    virtual void Process(){ // Start entry.
      rank = ps::MyRank();
      snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
      core_num = std::thread::hardware_concurrency();
      core_num = 1;
      std::cout<<"core_num = "<<core_num<<std::endl;
      batch_learning_threadpool();
      //online_learning_threadpool();
      //batch_learning_callback();
      std::cout<<"train end......"<<std::endl;
    }

  private:
    bool isStartToRun;
  public:
    int rank;
    int core_num;
    int batch_num;
    int block_size = 2;
    int epochs = 100;

    std::atomic_llong num_batch_fly = {0};
    std::atomic_llong all_time = {0};
    std::atomic_llong all_push_time = {0};
    std::atomic_llong all_pull_time = {0};
    std::atomic_llong send_key_numbers = {0};
    std::atomic_llong calculate_batch_gradient_thread_finish_num = {0};
    std::atomic_llong calculate_pctr_thread_finish_num = {0};

    float logloss = 0.0;
    float rmse = 0.0;
    std::vector<auc_key> auc_vec;
    std::vector<auc_key> test_auc_vec;

    std::ofstream md;
    std::mutex mutex;
    dml::Data *train_data;
    dml::Data *test_data;
    const char *train_file_path;
    const char *test_file_path;
    char train_data_path[1024];
    char test_data_path[1024];
    float bias = 0.0;
    ps::KVWorker<float> kv_;
};//end class worker

}//end namespace dmlc 
