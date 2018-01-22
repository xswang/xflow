/*
 * c_api.cc
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "src/c_api/c_api.h"

XF_DLL int XFCreate(void* *h,
                    const char* train_data_path,
                    const char* test_data_path) {
  XFlow* xf = new XFlow(train_data_path, test_data_path);
  *h = xf;
}

XF_DLL int XFStartTrain(void* *h) {
  XFlow* xf = reinterpret_cast<XFlow*>(*h);
  xf->lr_worker_->train();
}

