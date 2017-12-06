#include "worker.h"
#include "server.h"

#include "ps.h"

int main(int argc,char *argv[]){
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
