/*
 * c_api.cc
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "src/c_api/c_api.h"

XFLOW_DLL int XFlowCreate(XFLOW *xflow_api) {
  XFlow* xflow = new XFlow;
  xflow_api = xflow;
}
XFLOW_DLL int XFlowSetTrainAndTestDataPath(XFlow* xflow_api, const char* train_data_path,
                                           const char* test_data_path) {
  LRWorker lr_worker = new LRWorker(train_data_path, test_data_path);
} 



