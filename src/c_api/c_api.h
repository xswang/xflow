/*
 * c_api.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef C_API_H
#define C_API_H

#include "src/base/base.h"
#include "src/server.h"
#include "src/lr_worker.h"

#ifdef __cplusplus
#define XF_EXTERN_C extern "C"
#include <cstdio>
#else
#define XF_EXTERN_C
#include <stdio.h>
#include <stdint.h>
#endif

#define XF_DLL XF_EXTERN_C

XF_DLL int XFCreate(void* *h,
                    const char* train_path,
                    const char* test_path);
XF_DLL int XFStartTrain(void* *h);

class XFlow {
 public:
  XFlow(const char* train, const char* test) {
    lr_worker_ = new xflow::LRWorker(train, test);
  }
  ~XFlow() {
    delete lr_worker_;
  }
  
  xflow::LRWorker* lr_worker_;
};

#endif /* !C_API_H */
