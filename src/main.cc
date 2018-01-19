#include "src/lr_worker.h"
#include "src/fm_worker.h"
#include "src/ffm_worker.h"
#include "src/server.h"

#include "ps/ps.h"

int v_dim;

int main(int argc,char *argv[]){
  v_dim = 1;
  if (ps::IsServer()) {
    S* server = new S();
  }
  ps::Start();
  if (ps::IsWorker()) {
    LRWorker* lr_worker = new LRWorker(argv[1], argv[2]);
    lr_worker->train();
  }
  ps::Finalize();
}
