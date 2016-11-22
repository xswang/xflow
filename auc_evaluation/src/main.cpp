#include <stdlib.h>
#include "auc_cal.h"
#define MAX_FILENAME_LEN 4096

int main(int argc, char* argv[]){
    int rank = 0, nproc = 0;
    char pctr_file[MAX_FILENAME_LEN];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    snprintf(pctr_file, sizeof(pctr_file), "%s", argv[1]);
    
    Load_Data load_data;
    load_data.load_pctr_nclk_clk(pctr_file, rank);
    AUC auc(&load_data);
    auc.run(nproc, rank);
    
    MPI_Finalize();
    return 0;
}
