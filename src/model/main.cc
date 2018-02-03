/*
 * base.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "src/model/lr/lr_worker.h"
#include "src/model/fm/fm_worker.h"
#include "src/model/mvm/mvm_worker.h"
#include "src/model/server.h"

#include "ps/ps.h"

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cout << "sh run_ps_local.sh model_index epochs\n";
    std::cout << "LR model expmple: sh run_ps_local.sh 0 100\n";
    std::cout << "FM model expmple: sh run_ps_local.sh 1 100\n";
    std::cout << std::endl;
  }
  if (ps::IsServer()) {
    xflow::Server* server = new xflow::Server();
  }
  ps::Start();
  if (ps::IsWorker()) {
    int epochs = std::atoi(argv[4]);
    if (*(argv[3]) == '0') {
      std::cout << "start LR " << std::endl;
      xflow::LRWorker* lr_worker = new xflow::LRWorker(argv[1], argv[2]);
      lr_worker->epochs = epochs;
      lr_worker->train();
    }
    if (*(argv[3]) == '1') {
      std::cout << "start FM " << std::endl;
      xflow::FMWorker* fm_worker = new xflow::FMWorker(argv[1], argv[2]);
      fm_worker->epochs = epochs;
      fm_worker->train();
    }
    if (*(argv[3]) == '2') {
      std::cout<< "start MVM " << std::endl;
      xflow::MVMWorker* mvm_worker = new xflow::MVMWorker(argv[1], argv[2]);
      mvm_worker->epochs = epochs;
      mvm_worker->train();
    }
  }
  ps::Finalize();
}
