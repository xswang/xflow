#pragma once
#include "io.h"
#include <vector>
#include <set>
#include <string.h>

namespace dml{
class LoadData : public IO{
    public:
        LoadData(const char *file_path) : IO(file_path){}
        ~LoadData(){}

        void load_all_data();
        void load_minibatch_data(int num);
        void load_mibibatch_hash_data(int num);
	void load_minibatch_hash_data_fread(int bufsize);
    public:
        std::set<long int> feaIdx;
        std::set<long int>::iterator setIter;

        key_val keyval;
        std::vector<key_val> sample;
        std::vector<std::vector<key_val>> fea_matrix;
        std::vector<int> label;
};
}
