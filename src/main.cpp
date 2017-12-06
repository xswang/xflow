#include "worker.h"
#include "server.h"

#include "ps.h"

int main(int argc,char *argv[]){
  W* worker = new W(argv[1], argv[2]);
  S* server = new S();
  ps::Start();
  worker->Process();
  ps::Finalize();
}
