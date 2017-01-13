#include "ps.h"

namespace ps {
App* App::Create(int argc, char *argv[]) {//StartSystem->Postoffice::Run->manager_.Init->app_ = App::Create(argc, argv);
  App* app = new App();
  if (ps::IsServerNode()) {
    CreateServerNode(argc, argv);
  }
  return app;
}
}  // namespace ps

int main(int argc, char *argv[]) {
  ps::StartSystem(&argc, &argv);
  int ret = ps::IsWorkerNode() ? WorkerNodeMain(argc, argv) : 0;
  ps::StopSystem();
  return ret;
}
