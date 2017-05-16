#pragma once
#include <fstream>
#include <iostream>
#include <functional>
#include <vector>

namespace dml{

struct kv{
    int fgid;
    size_t fid;
    int val;
};

class IO{
    public:
        IO(const char *file_path) : file_path(file_path){
            Init();
        };
        ~IO(){};

        void Init(){
            fin_.open(file_path, std::ios::in);
            if(!fin_.is_open()){
                std::cout<<"open file "<<file_path<<" error! "<<std::endl;
                exit(1);
            }else{
                //std::cout<<"open file "<<file_path<<" sucess! "<<std::endl;
            }
            fp_ = fopen(file_path, "r"); 
        }

        virtual void load_all_data() = 0;
        virtual void load_minibatch_data(int num) = 0;

    public:
        std::ifstream fin_;
        FILE *fp_;
        std::string line;
        const char *file_path;

        int fgid;
        size_t fid;
        int val;
        char fid_str[1024];
        std::hash<std::string> h;

        size_t bmax = 0, btop = 0;
        int buffer_size = 0;
        int nchar;
        float y;
};

class Data
{
public:
    std::vector<std::vector<kv>> fea_matrix;
    std::vector<int> label;
};

}
