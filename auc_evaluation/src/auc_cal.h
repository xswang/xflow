#ifndef AUC_CALCULATE
#define AUC_CALCULATE

#include <string.h>
#include "mpi.h"
#include "load_data.h"

class AUC{

    public:
    AUC(Load_Data* ld) : data(ld){
        init();
    }
    ~AUC(){}

    void init(){
        auc = 0.0;
        glo_all_non_clk = new float[data->MAX_ARRAY_SIZE];
        glo_all_clk = new float[data->MAX_ARRAY_SIZE];
        glo_nclk = new float[data->MAX_ARRAY_SIZE];
        glo_clk = new float[data->MAX_ARRAY_SIZE];
    }

    void single_node_merge(){
        memset(glo_nclk, 0, data->MAX_ARRAY_SIZE * sizeof(float));
        memset(glo_clk, 0, data->MAX_ARRAY_SIZE * sizeof(float));
        int cnt = data->predict_list.size();
        for(int i = 0; i < cnt; i++){
            int idx = data->predict_list[i].idx;
            glo_nclk[idx] += data->predict_list[i].nclk;
            glo_clk[idx] += data->predict_list[i].clk;
        }
    }

    int auc_calculate(float* all_click, float* all_nclick, double& auc_res){
        double click_sum = 0.0;
        double nclick_sum = 0.0;
        double old_click_sum = 0.0;
        double clicksum_dot_nclicksum = 0.0;
        //double auc = 0.0;
        auc_res = 0.0;
        for(int i = 0; i < data->MAX_ARRAY_SIZE; i++){
            old_click_sum = click_sum;
            click_sum += all_click[i];
            nclick_sum += all_nclick[i];
            auc += (old_click_sum + click_sum) * all_nclick[i] / 2;
        }
        clicksum_dot_nclicksum = click_sum * nclick_sum;
        auc_res = auc/(clicksum_dot_nclicksum);
    }

    int mpi_auc(int nprocs, int rank, double &auc){
        MPI_Status status;
        if(rank != MASTER_ID){
            MPI_Send(glo_nclk, data->MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, MPI_NON_CLK_TAG, MPI_COMM_WORLD);
            MPI_Send(glo_clk, data->MAX_ARRAY_SIZE, MPI_FLOAT, MASTER_ID, MPI_CLK_TAG, MPI_COMM_WORLD);
        }
        else if(rank == MASTER_ID){
            for(int i = 0; i < data->MAX_ARRAY_SIZE; i++){
                glo_all_non_clk[i] = glo_nclk[i];
                glo_all_clk[i] = glo_clk[i];
            }
            for(int i = 1; i < nprocs; i++){
                MPI_Recv(glo_nclk, data->MAX_ARRAY_SIZE, MPI_FLOAT, i, MPI_NON_CLK_TAG, MPI_COMM_WORLD, &status);
                MPI_Recv(glo_clk, data->MAX_ARRAY_SIZE, MPI_FLOAT, i, MPI_CLK_TAG, MPI_COMM_WORLD, &status);
                for(int i = 0; i < data->MAX_ARRAY_SIZE; i++){
                    glo_all_non_clk[i] += glo_nclk[i];
                    glo_all_clk[i] += glo_clk[i];
                }
            }
            auc_calculate(glo_all_non_clk, glo_all_clk,auc);
        }
    }

    int run(int nprocs, int rank){
        FILE *fp = NULL;
        single_node_merge();
        mpi_auc(nprocs, rank, auc);

        if(MASTER_ID == rank){
            printf("AUC = %lf\n", auc);
        }
    }

    private:
    Load_Data* data;

    double auc;
    float* glo_all_non_clk;
    float* glo_all_clk;
    float* glo_nclk;
    float* glo_clk;
    double glo_total_clk;
    double glo_total_nclk;
};
#endif
