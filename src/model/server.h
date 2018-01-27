#include "iostream"
#include "ps/ps.h"
#include "src/optimizer/ftrl.h"
#include <time.h>

namespace xflow{
class Server{
 public:
  Server(){
    server_w_ = new ps::KVServer<float>(0);
    server_w_->set_request_handle(FTRL::KVServerFTRLHandle_w());
    server_v_ = new ps::KVServer<float>(1);
    server_v_->set_request_handle(FTRL::KVServerFTRLHandle_v());
    std::cout << "init server success " << std::endl;
  }
  ~Server(){}
  ps::KVServer<float>* server_w_;
  ps::KVServer<float>* server_v_;
};//end class Server
}
