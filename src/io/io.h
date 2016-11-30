#pragma once
#include <fstream>
#include <iostream>

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
                std::cout<<"open file "<<file_path<<" sucess! "<<std::endl;
            }
        }

        virtual void load_all_data() = 0;
        virtual void load_minibatch_data(int num) = 0;

    public:
        std::ifstream fin_;
        std::string line;
        typedef kv key_val;
        const char *file_path;

        int fgid;
        size_t fid;
        int val;

        int nchar;
        int y;
};
}
