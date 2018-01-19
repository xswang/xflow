#include "src/lr_worker.h"
#include "src/fm_worker.h"
#include "src/ffm_worker.h"
#include "src/server.h"

#include "ps/ps.h"
namespace xflow{
int v_dim;
}

int main(int argc,char *argv[]){
  xflow::v_dim = 1;
  if (ps::IsServer()) {
    xflow::Server* server = new xflow::Server();
  }
  ps::Start();
  if (ps::IsWorker()) {
    xflow::LRWorker* lr_worker = new xflow::LRWorker(argv[1], argv[2]);
    lr_worker->train();
  }
  ps::Finalize();
}
