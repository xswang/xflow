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
#include <pmmintrin.h>
#include <immintrin.h>
#include "sparsehash_memory/sparsepp.h"

#include "sparsehash_cpu/internal/sparseconfig.h"
#include "sparsehash_cpu/type_traits.h"
#include "sparsehash_cpu/sparsetable"
#include "hash_test_interface.h"

#include "./io/load_data.cc"
//#include "./io/load_data_from_kafka.cc"
#include "threadpool/thread_pool.h"
#include "ps.h"

#include <netdb.h>  
#include <net/if.h>  
#include <arpa/inet.h>  
#include <sys/ioctl.h>  
#include <sys/types.h>  
#include <sys/socket.h>  

#define IP_SIZE     16

namespace dmlc{
class Worker : public ps::App{
    public:
        Worker(const char *train_file, const char *test_file) : 
            train_file_path(train_file), test_file_path(test_file){ 
        }
        ~Worker(){
            delete train_data;
        } 
 
        virtual void ProcessRequest(ps::Message* request){
            //do nothing.
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
	    Process();
	}

        void predict(int rank){
            char buffer[1024];
            snprintf(buffer, 1024, "%d", rank);
            std::string filename = buffer;
            std::ofstream md;
            md.open("pred_" + filename + ".txt");
            if(!md.is_open()) std::cout<<"open pred file failure!"<<std::endl;
            std::cout<<"test_data size = "<<test_data->fea_matrix.size()<<std::endl;
            auto all_keys = std::vector<sample_key>();
            auto unique_keys = std::make_shared<std::vector<ps::Key>>();
            int line_num = 0;
            
            for(int row = 0; row < test_data->fea_matrix.size(); ++row) {
                int sample_size = test_data->fea_matrix[row].size();
		sample_key sk;
 		sk.sid = line_num;
                for(int j = 0; j < sample_size; ++j) {
                    size_t idx = h(test_data->fea_matrix[row][j].fid);
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
		int allkeys_fid = all_keys[j].fid;
		int weight_fid = (*unique_keys)[i];
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
                md<<pctr<<"\t"<<1 - test_data->label[i]<<"\t"<<test_data->label[i]<<std::endl;
	    }
           md.close();
        } 

        inline void filter_zero_element(std::vector<float>& gradient, std::vector<ps::Key>& nonzero_index, std::vector<float>& nonzero_gradient){
            for(int i = 0; i < init_index.size(); i++){
                int idx = init_index[i];
                float g = gradient[idx];
                if(g != 0.0){
                    nonzero_index.push_back(idx);
                    nonzero_gradient.push_back(g);
                }
            }
        }

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

        void zpush_callback(std::vector<sample_key> all_keys, std::shared_ptr<std::vector<ps::Key>> unique_keys, std::shared_ptr<std::vector<float>> w, int start, int end){
            auto wx = std::vector<float>(end - start + 1);
            for(int j = 0, i = 0; j < all_keys.size();){
                int allkeys_fid = all_keys[j].fid;
                int weight_fid = (*unique_keys)[i];
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
                int allkeys_fid = all_keys[j].fid;
                int gradient_fid = (*unique_keys)[i];
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
            size_t idx = 0; int value = 0; float pctr = 0;
            auto all_keys = std::vector<sample_key>();
            auto unique_keys = std::make_shared<std::vector<ps::Key>> ();;
            int line_num = 0;
            for(int row = start; row < end; ++row){
                int sample_size = train_data->fea_matrix[row].size();
                sample_key sk;
                sk.sid = line_num;
                for(int j = 0; j < sample_size; ++j){//for one instance
                    idx = h(train_data->fea_matrix[row][j].fid);
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
        }

        void batch_learning_callback(int core_num){
            /*
            train_data_kafka = new dml::LoadData_from_kafka();
            std::cout<<"start load kafka data"<<std::endl;
            train_data_kafka->load_data_from_kafka(10);
            std::cout<<"load kafka data end"<<std::endl;
            return;
            */
            train_data = new dml::LoadData(train_data_path);
            train_data->load_all_data();
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
        }

        void calculate_batch_gradient_threadpool(int start, int end){
            timespec all_start, all_end, all_elapsed_time;
            clock_gettime(CLOCK_MONOTONIC, &all_start);
            size_t idx = 0; int value = 0; float pctr = 0;
            auto all_keys = std::vector<sample_key>();
            auto unique_keys = std::make_shared<std::vector<ps::Key>> ();;
            int line_num = 0;
            for(int row = start; row < end; ++row){
                int sample_size = train_data->fea_matrix[row].size();
                sample_key sk;
                sk.sid = line_num;
                for(int j = 0; j < sample_size; ++j){//for one instance
                    idx = h(train_data->fea_matrix[row][j].fid);
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
            
            auto wx = std::vector<float>(end - start + 1);
            for(int j = 0, i = 0; j < all_keys.size();){
                int allkeys_fid = all_keys[j].fid;
                int weight_fid = (*unique_keys)[i];
                if(allkeys_fid == weight_fid){
                    wx[all_keys[j].sid] += (*w)[i];
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

            auto push_gradient = std::make_shared<std::vector<float> > (keys_size);
            for(int j = 0, i = 0; j < all_keys.size();){
                int allkeys_fid = all_keys[j].fid;
                int gradient_fid = (*unique_keys)[i];
                int sid = all_keys[j].sid;
                if(allkeys_fid == gradient_fid){
                    (*push_gradient)[i] += wx[sid];
                    ++j;
                }
                else if(allkeys_fid > gradient_fid){
                    ++i;
                }
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
        }

        void batch_learning_threadpool(int core_num){
            ThreadPool pool(core_num);
            timespec allstart, allend, allelapsed;
            for(int epoch = 0; epoch < epochs; ++epoch){
                train_data = new dml::LoadData(train_data_path);
                clock_gettime(CLOCK_MONOTONIC, &allstart);
                send_key_numbers = 0;
		int block = 0;
                while(1){
                    train_data->load_mibibatch_hash_data(block_size);
		    std::cout<<"block "<<block<<" size "<<block_size<<std::endl;
		    std::cout<<"train_data->fea_matrix.size() = "<<train_data->fea_matrix.size()<<std::endl;
                    if(train_data->fea_matrix.size() < block_size) break;
                    thread_size = train_data->fea_matrix.size() / core_num;
		    std::cout<<"thread_size = "<<thread_size<<std::endl;
                    for(int i = 0; i < core_num; ++i){
                        int start = i * thread_size;
                        int end = (i + 1)* thread_size;
			std::cout<<"start = "<<start<<" end = "<<end<<std::endl;
                        pool.enqueue(std::bind(&Worker::calculate_batch_gradient_threadpool, this, start, end));
                        //if(i == 0) usleep(4000);
                    }//end all batch
		    ++block;
                }
                clock_gettime(CLOCK_MONOTONIC, &allend);
                allelapsed = time_diff(allstart, allend);
                std::cout<<"rank "<<rank<<" per process : "<<train_data->fea_matrix.size() * 1e9 * 1.0 / (allelapsed.tv_sec * 1e9 + allelapsed.tv_nsec)<<std::endl;
                std::cout<<"rank "<<rank<<" send_key_number avage: "<<send_key_numbers * 1.0 / (batch_num * core_num)<<std::endl;
            }//end for all epoch
        }//end batch_learning_threadpool

        virtual void Process(){
            rank = ps::MyRank();
            
            snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
            core_num = std::thread::hardware_concurrency();
            batch_learning_threadpool(core_num);
            //batch_learning_callback(core_num);
            std::cout<<"train end......"<<std::endl;
        }

    public:
        int rank;
        int core_num = 32;
        int batch_num;
        int call_back = 1;
        int block_size = 32000;
        int thread_size;
        int epochs = 10;
        int calculate_gradient_thread_count;

        std::atomic_llong  num_batch_fly = {0};
        std::atomic_llong all_time = {0};
        std::atomic_llong all_push_time = {0};
        std::atomic_llong all_pull_time = {0};
        std::atomic_llong send_key_numbers = {0};
      
        std::hash<size_t> h;
        
        std::mutex mutex;
        std::vector<ps::Key> init_index;
        dml::LoadData *train_data;
        dml::LoadData *test_data;
        //dml::LoadData_from_kafka *train_data_kafka;
        const char *train_file_path;
        const char *test_file_path;
        char train_data_path[1024];
        char test_data_path[1024];
        float bias = 0.0;
        ps::KVWorker<float> kv_;
};//end class worker

}//end namespace dmlc 
