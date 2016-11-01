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

        void calculate_batch_gradient(int& start, int& end, std::vector<float>& w_all){
            long int idx = 0; int value = 0; float pctr = 0;
            std::vector<float> g(init_index.size());
            std::cout<<"start = "<<start<<" end = "<<end<<std::endl;
            for(int row = start; row < end; ++row){
                std::vector<ps::Key> keys;
                std::vector<int> values;
                if(rank == 0){
                for(int j = 0; j < train_data->fea_matrix[row].size(); ++j){//for one instance
                    idx = train_data->fea_matrix[row][j].fid;
                    keys.push_back(idx);
                    value = train_data->fea_matrix[row][j].val;
                    values.push_back(value);
                    if(rank == 0)std::cout<<""<<idx<<":"<<value<<" ";
                }
                std::cout<<std::endl;
                }
                float wx = bias;
                if(rank == 0) std::cout<<"-------------row-----------------"<<row<<std::endl; 
                for(int j = 0; j < keys.size(); j++){
                    //if(rank == 0)std::cout<<" keys = "<<j << " val = "<<keys[j]<<" "<<std::endl;
                    wx += w_all[keys[j]] * values[j];
                }
                std::cout<<std::endl;
                pctr = sigmoid(wx);
                float delta = pctr - train_data->label[row];
                for(int j = 0; j < keys.size(); j++){
                    g[keys[j]] += delta * values[j];
                }
                //std::cout<<"================================"<<std::endl;
            }
            kv_.Wait(kv_.Push(init_index, g));
        }

        virtual void Process(){
	        rank = ps::MyRank();
            snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);
            init_index.clear();
            for(int i = 0; i < 9e5; i++){
                init_index.push_back(i);
            }
            std::vector<float> init_val(9e5, 0.0);
            kv_.Wait(kv_.Push(init_index, init_val));

            core_num = std::thread::hardware_concurrency();
            core_num = 1;
            ThreadPool pool(core_num);

            for(int epoch = 0; epoch < epochs; ++epoch){
                train_data = new dml::LoadData(train_data_path);
                int batch = 0;
                while(1){
                    train_data->load_batch_data(batch_size);
                    for(int i = 0; i < train_data->fea_matrix.size(); ++i){
                        for(int j = 0; j < train_data->fea_matrix[0].size(); j++){
                            int idx = train_data->fea_matrix[i][j].fid;
                            int val = train_data->fea_matrix[i][j].val;
                            //std::cout<<idx<<"::"<<val<<" ";
                        }
                        //std::cout<<std::endl;
                    }
                    std::cout<<"batch size = "<<train_data->fea_matrix.size()<<std::endl;
                    if(train_data->fea_matrix.size() < batch_size){
                            std::cout<<"read all"<<std::endl;
                            break;
                    }
                    std::vector<float> w_all;
                    kv_.Wait(kv_.Pull(init_index, &w_all));
                    int thread_batch = batch_size / core_num;
                    for(int i = 0; i < core_num; ++i){
                        int start = i * thread_batch;
                        int end = (i + 1) * thread_batch;
                        //calculate_batch_gradient(start, end);
                        pool.enqueue(std::bind(&Worker::calculate_batch_gradient, this, start, end, w_all));
                    }//end for
                    sleep(1);
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
        int batch_size = 2;
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
        float alpha = 0.1;
        float beta = 1.0;
        float lambda1 = 0.001;
        float lambda2 = 0.0;
        int step = 2;
        ps::KVWorker<float> kv_;
};//end class worker

}//end namespace dmlc 
