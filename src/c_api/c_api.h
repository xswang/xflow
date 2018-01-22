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
#define XFLOW_EXTERN_C extern "C"
#include <cstdio>
#else
#define XFLOW_EXTERN_C
#include <stdio.h>
#include <stdint.h>
#endif

#define XFLOW_DLL XFLOW_EXTERN_C

XFLOW_DLL int XFlowCreate(void* h);

XFLOW_DLL int XFlowSetDataPath(void* h,
                               const char* train_path,
                               const char* test_path);
XFLOW_DLL int XFlowStartTrain(void* h);

class XFlow {
 public:
  XFlow() {}
  ~XFlow() {}
  
  xflow::LRWorker lr_worker_;
}；

#endif /* !C_API_H */
