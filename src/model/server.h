#include "iostream"
#include "ps/ps.h"
#include <time.h>

namespace xflow{
float alpha = 5e-2;
float beta = 1.0;
float lambda1 = 5e-5;
float lambda2 = 10.0;

int w_dim;
int v_dim;

typedef struct FTRLEntry_w{
  FTRLEntry_w(int k = w_dim) {
    w.resize(k, 0.0);
    n.resize(k, 0.0);
    z.resize(k, 0.0);
  };
  std::vector<float> w;
  std::vector<float> n;
  std::vector<float> z;
} ftrlentry_w;

struct KVServerFTRLHandle_w {
  void operator()(const ps::KVMeta& req_meta, 
                  const ps::KVPairs<float>& req_data, 
                  ps::KVServer<float>* server) {
    size_t keys_size = req_data.keys.size();
    size_t vals_size = req_data.vals.size();
    ps::KVPairs<float> res;

    if (req_meta.push) {
      w_dim = vals_size / keys_size;
      CHECK_EQ(keys_size, vals_size / w_dim);
    } else {
      res.keys = req_data.keys;
      res.vals.resize(keys_size);
    }

    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      FTRLEntry_w& val = store[key];
      for (int j = 0; j < w_dim; ++j){
        if (req_meta.push) {
          float g = req_data.vals[i * w_dim + j];
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
          for (int j = 0; j < w_dim; ++j) {
            res.vals[i * w_dim + j] = val.w[j];
          }
        }
      }
    }
    server->Response(req_meta, res);
  }
 private:
  std::unordered_map<ps::Key, ftrlentry_w> store;
};

typedef struct FTRLEntry_v{
  FTRLEntry_v(int k = v_dim) {
    w.resize(k, 0.001);
    n.resize(k, 0.0);
    z.resize(k, 0.0);
  };  
  std::vector<float> w;
  std::vector<float> n;
  std::vector<float> z;
} ftrlentry_v;

struct KVServerFTRLHandle_v {
  void operator()(const ps::KVMeta& req_meta,
      const ps::KVPairs<float>& req_data,
      ps::KVServer<float>* server) {
    size_t keys_size = req_data.keys.size();
    size_t vals_size = req_data.vals.size();
    ps::KVPairs<float> res;

    if (req_meta.push) {
      v_dim = vals_size / keys_size;
      CHECK_EQ(keys_size, vals_size / v_dim);
    } else {
      res.keys = req_data.keys;
      res.vals.resize(keys_size * v_dim);
    }

    for (size_t i = 0; i < keys_size; ++i) {
      ps::Key key = req_data.keys[i];
      FTRLEntry_v& val = store[key];
      for (int j = 0; j < v_dim; ++j){
        if (req_meta.push) {
          float g = req_data.vals[i * v_dim + j];
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
          for (int j = 0; j < v_dim; ++j) {
            res.vals[i * v_dim + j] = val.w[j];
          }
        }
      }
    }
    server->Response(req_meta, res);
  }
 private:
  std::unordered_map<ps::Key,ftrlentry_v> store;
}; 

class Server{
 public:
  Server(){
    server_w_ = new ps::KVServer<float>(0);
    server_w_->set_request_handle(KVServerFTRLHandle_w());
    server_v_ = new ps::KVServer<float>(1);
    server_v_->set_request_handle(KVServerFTRLHandle_v());
    std::cout << "init server success " << std::endl;
  }
  ~Server(){}
  ps::KVServer<float>* server_w_;
  ps::KVServer<float>* server_v_;
};//end class Server
}
