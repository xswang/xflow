#include <iostream>
#include <mutex>
#include <functional>
#include <time.h>
#include <unistd.h>
#include <memory>
#include "./io/load_data.cc"
#include "threadpool/thread_pool.h"
#include "ps.h"

namespace dmlc{
class Worker : public ps::App{
    public:
        Worker(const char *train_file, const char *test_file) : 
                train_file_path(train_file), test_file_path(test_file){ }
        ~Worker(){
            delete train_data;
            //delete test_data;
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

        void save_model(int epoch){
            char buffer[1024];
            snprintf(buffer, 1024, "%d", epoch);
            std::string filename = buffer;
            std::ofstream md;
            md.open("model/model_" + filename + ".txt");
            if(!md.is_open()) std::cout<<"save model open file error!"<<std::endl;

            std::vector<float> w_all;
            kv_.Wait(kv_.Pull(init_index, &w_all));
            for(int i = 0; i < init_index.size(); ++i){
                if(w_all[init_index[i]] != 0.0){
                    md << init_index[i]<<"\t"<<w_all[init_index[i]]<<std::endl;
                }
            }

            md.close();
        }
        
        void predict(int rank){
           char buffer[1024];
           snprintf(buffer, 1024, "%d", rank);
           std::string filename = buffer;
           std::ofstream md;
           md.open("pred_" + filename + ".txt");
           if(!md.is_open()) std::cout<<"open pred file failure!"<<std::endl;

           std::cout<<"test_data size = "<<test_data->fea_matrix.size()<<std::endl;
           std::vector<float> w_all;
           kv_.Wait(kv_.Pull(init_index, &w_all));
           for(int i = 0; i < test_data->fea_matrix.size(); i++) {
               float x = bias;
               for(int j = 0; j < test_data->fea_matrix[i].size(); j++) {
                   long int idx = test_data->fea_matrix[i][j].fid;
                   int value = test_data->fea_matrix[i][j].val;
                   x += w_all[idx] * value;
               }
               double pctr;
               if(x < -30){
                   pctr = 1e-6;
               }
               else if(x > 30){
                   pctr = 1.0;
               }
               else{
                   double ex = pow(2.718281828, x);
                   pctr = ex / (1.0 + ex);
               }
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

        void calculate_batch_gradient(int& start, int& end){
            timespec all_start, all_end, all_elapsed_time;
            clock_gettime(CLOCK_MONOTONIC, &all_start);

            size_t idx = 0; int value = 0; float pctr = 0;
            auto keys = std::make_shared<std::vector<ps::Key>> ();
            std::vector<float> w;
            for(int row = start; row < end; ++row){
                for(int j = 0; j < train_data->fea_matrix[row].size(); ++j){//for one instance
                    idx = train_data->fea_matrix[row][j].fid;
                    (*keys).push_back(idx);
                }
            }
            std::sort((*keys).begin(), (*keys).end());
            std::vector<ps::Key>::iterator iter_keys;
            iter_keys = unique((*keys).begin(), (*keys).end());
            (*keys).erase(iter_keys, (*keys).end());

            timespec pull_start_time, pull_end_time, pull_elapsed_time;
            clock_gettime(CLOCK_MONOTONIC, &pull_start_time);
            kv_.Wait(kv_.ZPull(keys, &w));
            clock_gettime(CLOCK_MONOTONIC, &pull_end_time);
            pull_elapsed_time = time_diff(pull_start_time, pull_end_time);
            
            std::map<size_t, float> weight;
            for(int i = 0; i < (*keys).size(); i++){
                weight.insert(std::pair<size_t, float>((*keys)[i], w[i]));
            }
            std::map<size_t, float> gradient;
            std::map<size_t, float>::iterator iter;

            for(int row = start; row < end; ++row){
                float wx = bias;
                for(int j = 0; j < train_data->fea_matrix[row].size(); ++j){
                    idx = train_data->fea_matrix[row][j].fid;
                    wx += weight[idx];
                }
                pctr = sigmoid(wx);
                float delta = pctr - train_data->label[row];
                for(int j = 0; j < (*keys).size(); j++){
                    gradient[(*keys)[j]] += delta;
                }
            }

            auto push_keys = std::make_shared<std::vector<ps::Key> > ();
            auto push_gradient = std::make_shared<std::vector<float> > ();
            for(iter = weight.begin(); iter != weight.end(); ++iter){
                (*push_keys).push_back(iter->first);
                (*push_gradient).push_back(gradient[iter->first]);
            }

            timespec push_start_time, push_end_time, push_elapsed_time;
            clock_gettime(CLOCK_MONOTONIC, &push_start_time);
            kv_.Wait(kv_.ZPush(push_keys, push_gradient));//put gradient to servers;
            clock_gettime(CLOCK_MONOTONIC, &push_end_time);
            push_elapsed_time = time_diff(push_start_time, push_end_time);

            clock_gettime(CLOCK_MONOTONIC, &all_end);
            all_elapsed_time = time_diff(all_start, all_end);

            all_time += all_elapsed_time.tv_sec * 1e9 + all_elapsed_time.tv_nsec; 
            all_pull_time += pull_elapsed_time.tv_sec * 1e9 + pull_elapsed_time.tv_nsec;
            all_push_time += push_elapsed_time.tv_sec * 1e9 + push_elapsed_time.tv_nsec;
            send_key_numbers += (*keys).size();
        }

        void online_learning(int core_num){
            ThreadPool pool(core_num);
            train_data = new dml::LoadData(train_data_path);
            int batch = 0, start, end, thread_batch = batch_size / core_num;
            while(1){
                train_data->load_minibatch_data(batch_size);
                if(train_data->fea_matrix.size() < batch_size){
                    std::cout<<"read all"<<std::endl;
                    break;
                }
                calculate_gradient_thread_count = 0;

                for(int i = 0; i < core_num; ++i){
                    start = i * thread_batch;
                    end = (i + 1) * thread_batch;
                    pool.enqueue(std::bind(&Worker::calculate_batch_gradient, this, start, end));
                }//end for
                while(calculate_gradient_thread_count < core_num);
                if((batch + 1) % 20 == 0)std::cout<<"rank "<<rank<<" batch = "<<batch<<std::endl;
                ++batch;
            }//end while one epoch
        }

        void batch_learning(int core_num){
            train_data = new dml::LoadData(train_data_path);
            train_data->load_all_data();
            std::cout<<"train_data size : "<<train_data->fea_matrix.size()<<std::endl;

            ThreadPool pool(core_num);

            batch_num = train_data->fea_matrix.size() / batch_size;
            std::cout<<"batch_num : "<<batch_num<<std::endl;
            for(int epoch = 0; epoch < epochs; ++epoch){
                size_t old_all_time = 0;
                size_t old_all_push_time = 0;
                size_t old_all_pull_time = 0;
                for(int i = 0; i < batch_num; ++i){
                    if((i + 1)%300 == 0){
                        std::cout<<"rank "<<rank<<" epoch "<<epoch<<" batch "<<i<<std::endl;
                        std::cout<<"rank "<<rank<<" all time avage: "<<(all_time - old_all_time)* 1.0 / (300 * core_num) <<std::endl;
                        std::cout<<"rank "<<rank<<" all push time avage: "<<(all_push_time - old_all_push_time) * 1.0 / (300 * core_num)<<std::endl;
                        std::cout<<"rank "<<rank<<" all pull time avage: "<<(all_pull_time - old_all_pull_time) * 1.0 / (300 * core_num)<<std::endl;
                        old_all_time = all_time;
                        old_all_push_time = all_push_time;
                        old_all_pull_time = all_pull_time;
                    }
                    int all_start = i * batch_size;
                    int thread_batch = batch_size / core_num;
                    int start, end;
                    for(int j = 0; j < core_num; ++j){
                        start = all_start + j * thread_batch;
                        end = all_start + (j + 1) * thread_batch;
                        pool.enqueue(std::bind(&Worker::calculate_batch_gradient, this, start, end));
                    }
                }//end all batch
                std::cout<<"rank "<<rank<<" all time avage: "<<all_time * 1.0 / (batch_num * core_num) <<std::endl;
                std::cout<<"rank "<<rank<<" all push time avage: "<<all_push_time * 1.0 / (batch_num * core_num)<<std::endl;
                std::cout<<"rank "<<rank<<" all pull time avage: "<<all_pull_time * 1.0 / (batch_num * core_num)<<std::endl;
                std::cout<<"rank "<<rank<<" send_key_number avage: "<<send_key_numbers * 1.0 / (batch_num * core_num)<<std::endl;
            }//end all epoch
        }

        virtual void Process(){
            rank = ps::MyRank();
            snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);

            core_num = std::thread::hardware_concurrency();

            if(is_online_learning == 1){
                online_learning(core_num);
            }
            else if(is_batch_learning == 1){
                batch_learning(core_num);
            }
            std::cout<<"train end......"<<std::endl;
            /*
            snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
            test_data = new dml::LoadData(test_data_path);
            test_data->load_all_data();
            predict(rank);
            std::cout<<"rank "<<rank<<" end!"<<std::endl;
            */
        }//end process

    public:
        int rank;
        int core_num;
        int batch_num;
        int batch_size = 400;
        int epochs = 1;
        int calculate_gradient_thread_count;
        int is_online_learning = 0;
        int is_batch_learning = 1;

        std::atomic_llong all_time = {0};
        std::atomic_llong all_push_time = {0};
        std::atomic_llong all_pull_time = {0};
        std::atomic_llong send_key_numbers = {0};
        
        std::mutex mutex;
        std::vector<ps::Key> init_index;
        dml::LoadData *train_data;
        dml::LoadData *test_data;
        const char *train_file_path;
        const char *test_file_path;
        char train_data_path[1024];
        char test_data_path[1024];
        float bias = 0.0;
        ps::KVWorker<float> kv_;
};//end class worker

}//end namespace dmlc 
