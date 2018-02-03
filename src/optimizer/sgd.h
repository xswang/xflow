/*
 * sgd.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SRC_OPTIMIZER_SGD_H_
#define SRC_OPTIMIZER_SGD_H_

#include <vector>

namespace xflow {
extern int w_dim;
extern int v_dim;
float learning_rate = 0.001;

class SGD {
 public:
  SGD() {}
  ~SGD() {}

  typedef struct SGDEntry_w {
    SGDEntry_w(int k = w_dim) {
      w.resize(k, 0.0);
    }
    std::vector<float> w;
  } sgdentry_w;

  struct KVServerSGDHandle_w {
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
        res.vals.resize(keys_size * w_dim);
      }

      for (size_t i = 0; i < keys_size; ++i) {
        ps::Key key = req_data.keys[i];
        SGDEntry_w& val = store[key];
        for (int j = 0; j < w_dim; ++j) {
          if (req_meta.push) {
            float g = req_data.vals[i * w_dim + j];
            val.w[j] -= learning_rate * g;
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
    std::unordered_map<ps::Key, sgdentry_w> store;
  };

  typedef struct SGDEntry_v {
    SGDEntry_v(int k = v_dim) {
      w.resize(k, 0.001);
    }
    std::vector<float> w;
  } sgdentry_v;

  struct KVServerSGDHandle_v {
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
        SGDEntry_v& val = store[key];
        for (int j = 0; j < v_dim; ++j) {
          if (req_meta.push) {
            float g = req_data.vals[i * v_dim + j];
            val.w[j] -= learning_rate * g;
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
    std::unordered_map<ps::Key, sgdentry_v> store;
  };

 private:
};
}  // namespace xflow

#endif  // SRC_OPTIMIZER_SGD_H_
