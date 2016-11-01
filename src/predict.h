#ifndef PREDICT_H_
#define PREDICT_H_

#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include "load_data.h"

typedef struct{
    float clk;
    float nclk;
    long idx;
} clkinfo;

class Predict{
    public:
    Predict(Load_Data* load_data) 
            : data(load_data){
        pctr = 0.0;
        MAX_ARRAY_SIZE = 2000;
        g_all_non_clk = new float[MAX_ARRAY_SIZE];
        g_all_clk = new float[MAX_ARRAY_SIZE];
        g_nclk = new float[MAX_ARRAY_SIZE];
        g_clk = new float[MAX_ARRAY_SIZE];
    }
    ~Predict(){
        delete[] g_all_non_clk;
        delete[] g_all_clk;
        delete[] g_nclk;
        delete[] g_clk;
    }

    //void predict(std::vector<float> glo_w){
    void predict(std::vector<float> glo_w){
        std::cout<<"glo_w size "<<glo_w.size()<<std::endl;
        for(int i = 0; i < data->fea_matrix.size(); i++) {
	        float x = 0.0;
            for(int j = 0; j < data->fea_matrix[i].size(); j++) {
                long index = data->fea_matrix[i][j].idx;
                int value = data->fea_matrix[i][j].val;
                x += glo_w[index] * value;
            }

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
            int id = int(pctr*MAX_ARRAY_SIZE);
            clkinfo clickinfo;
            clickinfo.clk = data->label[i];
            clickinfo.nclk = 1 - data->label[i];
            clickinfo.idx = id;
            result_list.push_back(clickinfo);
        }
    }

    void merge_clk(){//merge local node`s clk
        memset(g_nclk, 0.0, MAX_ARRAY_SIZE * sizeof(float));
        memset(g_clk, 0.0, MAX_ARRAY_SIZE * sizeof(float));
        int cnt = result_list.size();
        for(int i = 0; i < cnt; i++){
            long index = result_list[i].idx;
            g_nclk[index] += result_list[i].nclk;
            g_clk[index] += result_list[i].clk;
        }
    }

    int auc_cal(float* all_clk, float* all_nclk, double& auc_res){
            double clk_sum = 0.0;
            double nclk_sum = 0.0;
            double old_clk_sum = 0.0;
            double clksum_multi_nclksum = 0.0;
            auc_res = 0.0;
            for(int i = 0; i < MAX_ARRAY_SIZE; i++){
                    old_clk_sum = clk_sum;
                    clk_sum += all_clk[i];
                    nclk_sum += all_nclk[i];
                    auc += (old_clk_sum + clk_sum) * all_nclk[i] / 2;
            }
            clksum_multi_nclksum = clk_sum * nclk_sum;
            auc_res = auc/(clksum_multi_nclksum);
    }

    //void run(std::vector<float> w){
    void run(std::vector<float> &w){
        predict(w);

        merge_clk();

        printf("AUC = %lf\n", auc);
    }

    private:
    Load_Data* data;
    std::vector<clkinfo> result_list;
    int MAX_ARRAY_SIZE;
    double auc = 0.0;
    float* g_all_non_clk;
    float* g_all_clk;
    float* g_nclk;
    float* g_clk;
    float g_total_clk;
    float g_total_nclk;

    float pctr;
    int nproc; // total num of process in MPI comm world
    int rank; // my process rank in MPT comm world
};
#endif
