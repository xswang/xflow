#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <map>
#include <set>
#include <vector>
#include <string.h>
#include <math.h>

#define TAB "\t"
#define CTRL_B "^B"

#define MASTER_ID (0)
#define MPI_NON_CLK_TAG (0)
#define MPI_CLK_TAG (1)

typedef struct{
    double nclk;
    double clk;
    long int idx;
}predinfo;

class Load_Data{
    public:
    Load_Data(){
        init();
    }
    ~Load_Data(){}

    void init(){
        pctr = 0.0;
        nclk = 0.0;
        clk = 0.0;
        MAX_ARRAY_SIZE = 1000;
    } 

    int load_pctr_nclk_clk(const char* str_ins_path, int rank){
        std::ifstream ifs;
        std::string line = "";
        std::string tmpstr = "";
        char ins_path[2048];
        snprintf(ins_path, 2048, "%s-%05d", str_ins_path, rank);
        ifs.open(ins_path);
        predict_list.clear();
        if (!ifs.is_open()) {
            std::cout<<"open "<<ins_path<<" failure! "<<std::endl;
        }

        while(getline(ifs, line)){
            int pos = line.find(CTRL_B);
            if(pos <= 0) tmpstr = line;
            else tmpstr = line.substr(0, pos);
            pos = tmpstr.find(TAB);
            pctr = atof(tmpstr.substr(0, pos).c_str());
            tmpstr = tmpstr.substr(pos+1, tmpstr.size() - pos -1);
            pos = tmpstr.find(TAB);
            nclk = atof(tmpstr.substr(0, pos).c_str());
            tmpstr = tmpstr.substr(pos+1, tmpstr.size() - pos -1);
            pos = tmpstr.find(CTRL_B);
            clk = atof(tmpstr.substr(0, pos).c_str());  

            int id = int(pctr*MAX_ARRAY_SIZE);
            predinfo predictinfo;
            predictinfo.nclk = nclk;
            predictinfo.clk = clk;
            predictinfo.idx = id;
            predict_list.push_back(predictinfo);
        }//end while
    }
    public:
    int MAX_ARRAY_SIZE;
    std::vector<predinfo> predict_list;
    private:
    double pctr;
    double nclk;
    double clk;
};
