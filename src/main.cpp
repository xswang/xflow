#include "worker.h"
#include "server.h"

#include "ps.h"

int v_dim;

int main(int argc,char *argv[]){
  v_dim = 6;
  if (ps::IsServer()) {
    S* server = new S();
  }
  ps::Start();
  if (ps::IsWorker()) {
    W* worker = new W(argv[1], argv[2]);
    worker->P();
  }
  ps::Finalize();
}
