#pragma once
#include "io.h"
#include <vector>
#include <set>
#include <string.h>

namespace dml{

class LoadData : public IO{
    public:
        LoadData(const char *file_path, size_t block_size) : IO(file_path), buf(block_size){}
        ~LoadData(){}

        void load_all_data();
        void load_all_hash_data();
        void load_minibatch_data(int num);
        void load_mibibatch_hash_data(int num);
	    void load_minibatch_hash_data_fread();
    private:
        std::vector<char> buf;
    public:
        Data m_data;
};

}
