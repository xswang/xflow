#include <iostream>
#include <mutex>
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
               float x = 0.0;
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

        void calculate_batch_gradient(int& start, int& end, std::vector<float>& w_all){
            long int idx = 0; int value = 0; float pctr = 0;
            std::vector<float> g(init_index.size());
            for(int row = start; row < end; ++row){
                std::vector<ps::Key> keys;
                std::vector<int> values;
                for(int j = 0; j < train_data->fea_matrix[row].size(); ++j){//for one instance
                    idx = train_data->fea_matrix[row][j].fid;
                    keys.push_back(idx);
                    value = train_data->fea_matrix[row][j].val;
                    values.push_back(value);
                }
                float wx = bias;
                for(int j = 0; j < keys.size(); j++){
                    wx += w_all[keys[j]] * values[j];
                }
                pctr = sigmoid(wx);
                float delta = pctr - train_data->label[row];
                for(int j = 0; j < keys.size(); j++){
                    g[keys[j]] += delta * values[j];
                }
            }
            std::vector<ps::Key> nonzero_index;
            std::vector<float> nonzero_gradient;
            filter_zero_element(g, nonzero_index, nonzero_gradient);
            kv_.Wait(kv_.Push(nonzero_index, nonzero_gradient));//put gradient to servers;
        }

        virtual void Process(){
	        rank = ps::MyRank();
            snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
            init_index.clear();
            for(int i = 0; i < 2e6; i++){
                init_index.push_back(i);
            }
            std::vector<float> init_val(2e6, 0.0);
            kv_.Wait(kv_.Push(init_index, init_val));

            core_num = std::thread::hardware_concurrency();
            //core_num = 1;
            ThreadPool pool(core_num);

            for(int epoch = 0; epoch < epochs; ++epoch){
                train_data = new dml::LoadData(train_data_path);
                int batch = 0;
                while(1){
                    train_data->load_batch_data(batch_size);
                    std::cout<<"batch size = "<<train_data->fea_matrix.size()<<std::endl;
                    if(train_data->fea_matrix.size() < batch_size){
                        std::cout<<"read all"<<std::endl;
                        break;
                    }
                    std::vector<float> w_all;
                    kv_.Wait(kv_.Pull(init_index, &w_all));//get weight from servers
                    int start, end;
                    int thread_batch = batch_size / core_num;
                    for(int i = 0; i < core_num; ++i){
                        start = i * thread_batch;
                        end = (i + 1) * thread_batch;
                        pool.enqueue(std::bind(&Worker::calculate_batch_gradient, this, start, end, w_all));
                    }//end for
                    calculate_batch_gradient(start, end, w_all);
                    std::cout<<"rank "<<rank<<" batch = "<<batch<<std::endl;
                    ++batch;
                }//end while

                if(rank == 0){
                    save_model(epoch);
                }
            }//end for

            snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
            test_data = new dml::LoadData(test_data_path);
            test_data->load_all_data();
            std::cout<<"rank "<<rank<<" end!"<<std::endl;
            predict(rank);
        }//end process

    public:
        int a = 0;
        int core_num;
        int batch_size = 200;
        int epochs = 1;

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
        float alpha = 2.0;
        float beta = 1.0;
        float lambda1 = 5.0;
        float lambda2 = 0.0;
        ps::KVWorker<float> kv_;
};//end class worker

}//end namespace dmlc 
