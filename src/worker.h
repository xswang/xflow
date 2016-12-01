#include <iostream>
#include <mutex>
#include <functional>
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

        //void calculate_batch_gradient(int& start, int& end, std::vector<float> &w_all){
        void calculate_batch_gradient(int& start, int& end){
            size_t idx = 0; int value = 0; float pctr = 0;
            std::vector<ps::Key> keys;
            std::vector<float> w;
            for(int row = start; row < end; ++row){
                for(int j = 0; j < train_data->fea_matrix[row].size(); ++j){//for one instance
                    idx = train_data->fea_matrix[row][j].fid;
                    keys.push_back(idx);
                }
            }
            std::sort(keys.begin(), keys.end());
            kv_.Wait(kv_.Pull(keys, &w));

            std::map<size_t, float> weight;
            for(int i = 0; i < keys.size(); i++){
                weight.insert(std::pair<size_t, float>(keys[i], w[i]));
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
                for(int j = 0; j < keys.size(); j++){
                    iter = gradient.find(keys[j]);
                    if(iter == gradient.end()){
                        float g = iter->second + delta;
                        gradient.insert(std::pair<size_t, float>(keys[j], g));
                    }
                    else{
                        float g = iter->second + delta;
                        gradient[keys[j]] = delta;
                    }
                }
            }

            std::vector<ps::Key> push_keys;
            std::vector<float> push_gradient;
            for(iter = weight.begin(); iter != weight.end(); ++iter){
                push_keys.push_back(iter->first);
                push_gradient.push_back(gradient[iter->first]);
            }

            kv_.Wait(kv_.Push(push_keys, push_gradient));//put gradient to servers;

            mutex.lock();
            calculate_gradient_thread_count++;
            mutex.unlock();
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

            snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
            test_data = new dml::LoadData(test_data_path);
            test_data->load_all_data();
            predict(rank);
            std::cout<<"rank "<<rank<<" end!"<<std::endl;
        }

        void batch_learning(int core_num){
            train_data = new dml::LoadData(train_data_path);
            train_data->load_all_data();
            std::cout<<"train_data size : "<<train_data->fea_matrix.size()<<std::endl;

            ThreadPool pool(core_num);

            batch_num = train_data->fea_matrix.size() / batch_size;
            std::cout<<"batch_num : "<<batch_num<<std::endl;

            for(int epoch = 0; epoch < epochs; ++epoch){
                for(int i = 0; i < batch_num; ++i){
                    if((batch_num + 1)%30 == 0) std::cout<<"rank "<<rank<<" epoch "<<epoch<<" batch "<<i<<std::endl;
                    int all_start = i * batch_size;
                    int thread_batch = batch_size / core_num;
                    int start, end;
                    calculate_gradient_thread_count = 0;

                    for(int j = 0; j < core_num; ++j){
                        start = all_start + j * thread_batch;
                        end = all_start + (j + 1) * thread_batch;
                        pool.enqueue(std::bind(&Worker::calculate_batch_gradient, this, start, end));
                    }
                    while(calculate_gradient_thread_count < core_num);//m
                }
            }
            snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
            test_data = new dml::LoadData(test_data_path);
            test_data->load_all_data();
            predict(rank);
            std::cout<<"rank "<<rank<<" end!"<<std::endl;
        }

        virtual void Process(){
            rank = ps::MyRank();
            snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);

            core_num = std::thread::hardware_concurrency();

            if(is_online_learning == true){
                batch_learning(core_num);
            }
            else if(is_batch_learning == true){
                online_learning(core_num);
            }
        }//end process

    public:
        int core_num;
        int batch_num;
        int batch_size = 400;
        int epochs = 100;
        int calculate_gradient_thread_count;
        bool is_online_learning = false;
        bool is_batch_learning = true;

        std::mutex mutex;
        std::vector<ps::Key> init_index;
        dml::LoadData *train_data;
        dml::LoadData *test_data;
        const char *train_file_path;
        const char *test_file_path;
        char train_data_path[1024];
        char test_data_path[1024];
        int rank;
        float bias = 0.0;
        ps::KVWorker<float> kv_;
};//end class worker

}//end namespace dmlc 
