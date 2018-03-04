/*
 * server.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SRC_MODEL_SERVER_H_
#define SRC_MODEL_SERVER_H_

#include <time.h>

#include <iostream>

#include "ps/ps.h"
#include "src/optimizer/ftrl.h"
#include "src/optimizer/sgd.h"

namespace xflow {
class Server {
 public:
  Server() {
    server_w_ = new ps::KVServer<float>(0);
    server_w_->set_request_handle(FTRL::KVServerFTRLHandle_w());
    //server_w_->set_request_handle(SGD::KVServerSGDHandle_w());

    server_v_ = new ps::KVServer<float>(1);
    server_v_->set_request_handle(FTRL::KVServerFTRLHandle_v());
    //server_v_->set_request_handle(SGD::KVServerSGDHandle_v());
    std::cout << "init server success " << std::endl;
  }
  ~Server() {}
  ps::KVServer<float>* server_w_;
  ps::KVServer<float>* server_v_;
};
}  // namespace xflow
#endif  // SRC_MODEL_SERVER_H_
