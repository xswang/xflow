/*
 * base.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef BASE_H
#define BASE_H
namespace xflow{
class Base{
 public:
  Base() {}
  ~Base() {}

  float sigmoid(float x) {
    if(x < -30) return 1e-6;
    else if(x > 30) return 1.0;
    else{
      double ex = pow(2.718281828, x);
      return ex / (1.0 + ex);
    }
  }
};

}
#endif /* !BASE_H */
