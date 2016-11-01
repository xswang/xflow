#include <iostream>
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
        void save_model(int st){
            char buffer[1024];
            snprintf(buffer, 1024, "%d", st);
            std::string filename = buffer;
            std::ofstream md;
            md.open("model/model_" + filename + ".txt");
            if(!md.is_open()) std::cout<<"save model open file error!"<<std::endl;
            std::cout<<"feaIdx size = "<<train_data->feaIdx.size()<<std::endl;
            for(int i = 0; i < init_index.size(); i++){
                if(w_all[init_index[i]] != 0.0) md << init_index[i]<<"\t"<<w_all[init_index[i]]<<std::endl;
            }
            md.close();
        }
        
        void predict(int r){
           char buffer[1024];
           snprintf(buffer, 1024, "%d", r);
           std::string filename = buffer;
           std::ofstream md;
           md.open("pred_" + filename + ".txt");
           if(!md.is_open()) std::cout<<"open pred file failure!"<<std::endl;
           std::cout<<"test_data size = "<<test_data->fea_matrix.size()<<std::endl;
           for(int i = 0; i < test_data->fea_matrix.size(); i++) {
               float x = 0.0;
               for(int j = 0; j < test_data->fea_matrix[i].size(); j++) {
                   long index = test_data->fea_matrix[i][j].fid;
                   int value = test_data->fea_matrix[i][j].val;
                   x += w_all[index] * value;
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

        void batch_gradient_calculate(int& start, int& end){
            int index = 0; float value = 0.0; float pctr = 0;
            kv_.Wait(kv_.Pull(init_index, &w_all));
            std::vector<float> g(init_index.size());
            for(int row = start; row < end; ++row){
                std::vector<ps::Key> keys;
                std::vector<float> values;
                //std::vector<float> w;
                for(int j = 0; j < train_data->fea_matrix[row].size(); j++){//for one instance
                    index = train_data->fea_matrix[row][j].fid;
                    keys.push_back(index);
                    value = train_data->fea_matrix[row][j].val;
                    values.push_back(value);
                }
                //kv_.Wait(kv_.Pull(keys, &w));
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
            kv_.Wait(kv_.Push(init_index, g));
        }

        virtual void Process(){
	        rank = ps::MyRank();
            snprintf(train_data_path, 1024, "%s-%05d", train_file_path, rank);

            init_index.clear();
            for(int i = 0; i < 3e6; i++){
                init_index.push_back(i);
            }
            std::vector<float> init_val(3e6, 0.0);
            kv_.Wait(kv_.Push(init_index, init_val));

            for(int epoch = 0; epoch < epochs; ++epoch){
                train_data = new dml::LoadData(train_data_path);
                while(1){
                    train_data->load_batch_data(batch_size);
                    if(train_data->fea_matrix.size() < batch_size) break;
                    int thread_batch = batch_size / core_num;
                    for(int i = 0; i < core_num; ++i){
                        int start = i * thread_batch;
                        int end = (i + 1) * thread_batch;
                        batch_gradient_calculate(start, end);
                    }
                }//end for minibatch
                if(rank == 0){
                    std::cout<<"end"<<std::endl;
                    save_model(epoch);
                }
            }//end for

            snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
            std::cout<<" test data_path======================"<<test_data_path<<std::endl;
            test_data = new dml::LoadData(test_data_path);
            test_data->load_all_data();
            predict(rank);
        }//end process

    public:
        int core_num;
        int batch_size = 200;
        int epochs = 5;

        std::vector<ps::Key> init_index;
        std::vector<ps::Key> fea_all;
        std::vector<float> w_all;	
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
