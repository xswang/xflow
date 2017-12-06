#include "iostream"
#include "ps.h"
#include <time.h>

class Scheduler{
  public:
    Scheduler(){}
    ~Scheduler(){}
};

float alpha;
float beta;
float lambda1;
float lambda2;

struct FTRLEntry{
  float w = 0.0;
  float z = 0.0;
  float n = 0.0;
};

struct KVServerFTRLHandle {
  void operator()(const ps::KVMeta& req_meta, const ps::KVPairs<float>& req_data, ps::KVServer<float>* server) {
    size_t n = req_data.keys.size();
    ps::KVPairs<float> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys;
      res.vals.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      ps::Key key = req_data.keys[i];
      if (req_meta.push) {
        float g = req_data.vals[i];
        float old_n = store[key].n;
        float n = old_n + g * g;
        store[key].z += g - (std::sqrt(n) - std::sqrt(old_n)) / alpha * store[key].w;
        store[key].n = n;
        if (std::abs(store[key].z) <= lambda1) {
          store[key].w = 0.0;
        } else {
          float tmpr= 0.0;
          if (store[key].z >= 0.0) tmpr = store[key].z - lambda1;
          else tmpr = store[key].z + lambda1;
          float tmpl = -1 * ( (beta + store[key].n)/alpha  + lambda2 );
          store[key].w = tmpr / tmpl;
        }
      } else {
        res.vals[i] = store[key].w;
      }
    }
    server->Response(req_meta, res);
  }
 private:
  std::unordered_map<ps::Key, FTRLEntry> store;
};

class S{
 public:
  S(){
    auto server_ = new ps::KVServer<float>(0);
    server_->set_request_handle(KVServerFTRLHandle());
  }
  ~S(){}
  ps::KVServer<float>* server_;
};//end class Server
