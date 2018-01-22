/*
 * c_api.cc
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "src/c_api/c_api.h"

XFLOW_DLL int XFlowCreate(void* *h) {
  XFlow* xf = new XFlow;
  *h = xf;
}
XFLOW_DLL int XFlowSetDataPath(void* *h,
                               const char* train_data_path,
                               const char* test_data_path) {
  XFlow* xf = reinterpret_cast<XFlow*>(*h);
  xf->lr_worker_ = new xflow::LRWorker(train_data_path, test_data_path);
}

XFLOW_DLL int XFlowStartTrain(void* h) {
  XFlow* xf = reinterpret_cast<XFlow*>(*h);
  xf->lr_worker_->train();
}

