#include "iostream"
#include "ps.h"
#include <time.h>

float alpha = 5e-2;
float beta = 1.0;
float lambda1 = 5e-5;
float lambda2 = 15.0;

typedef struct FTRLEntry{
  FTRLEntry(int k = 10) {
    w.resize(k, 0.0);
    n.resize(k, 0.0);
    z.resize(k, 0.0);
  }
  std::vector<float> w;
  std::vector<float> n;
  std::vector<float> z;
} ftrlentry;

struct KVServerFTRLHandle {
  void operator()(const ps::KVMeta& req_meta, const ps::KVPairs<float>& req_data, ps::KVServer<float>* server) {
    int k = 1;
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
      FTRLEntry& val = store[key];
      for (int j = 0; j < 1; ++j){
        if (req_meta.push) {
          float g = req_data.vals[i * k + j];
          float old_n = val.n[j];
          float n = old_n + g * g;
          val.z[j] += g - (std::sqrt(n) - std::sqrt(old_n)) / alpha * val.w[j];
          val.n[j] = n;
          if (std::abs(val.z[j]) <= lambda1) {
            val.w[j] = 0.0;
          } else {
            float tmpr= 0.0;
            if (val.z[j] > 0.0) tmpr = val.z[j] - lambda1;
            if (val.z[j] < 0.0) tmpr = val.z[j] + lambda1;
            float tmpl = -1 * ( (beta + std::sqrt(val.n[j]))/alpha  + lambda2 );
            val.w[j] = tmpr / tmpl;
          }
        } else {
          for (int j = 0; j < 1; ++j) {
            res.vals[i * k + j] = val.w[j];
          }
        }
      }
    }
    server->Response(req_meta, res);
  }
 private:
  //std::unordered_map<ps::Key, FTRLEntry> store;
  std::unordered_map<ps::Key, ftrlentry> store;
};

class S{
 public:
  S(){
    auto server_ = new ps::KVServer<float>(0);
    server_->set_request_handle(KVServerFTRLHandle());
    std::cout << "init server success " << std::endl;
  }
  ~S(){}
  ps::KVServer<float>* server_;
};//end class Server
