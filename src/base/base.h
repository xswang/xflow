/*
 * base.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef BASE_H
#define BASE_H

#include <stddef.h>
#include <vector>

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

  struct sample_key{
    size_t fid;
    int sid;
  };

  static bool sort_finder(const sample_key& a, const sample_key& b){
    return a.fid < b.fid;
  }

  static bool unique_finder(const sample_key& a, const sample_key& b){
    return a.fid == b.fid;
  }

  struct auc_key{
    int label;
    float pctr;
  };

  void calculate_auc(std::vector<auc_key>& auc_vec){
    std::sort(auc_vec.begin(), auc_vec.end(), [](const auc_key& a, const auc_key& b){
      return a.pctr > b.pctr;
    });
    float area = 0.0;
    int tp_n = 0;
    for(size_t i = 0; i < auc_vec.size(); ++i){
      if(auc_vec[i].label == 1) tp_n += 1;
      else area += tp_n;
      logloss += auc_vec[i].label * std::log2(auc_vec[i].pctr)+
        + (1.0 - auc_vec[i].label) * std::log2(1.0 - auc_vec[i].pctr);
    }
    logloss /= auc_vec.size();
    std::cout << "logloss: " << logloss << "\t";
    if (tp_n == 0 || tp_n == auc_vec.size()) std::cout<<"tp_n = "<<tp_n<<std::endl;
    else{
      area /= 1.0 * (tp_n * (auc_vec.size() - tp_n));
      std::cout<<"auc = "<<area<<"\ttp = "<<tp_n<<" fp = "<<(auc_vec.size() - tp_n)<<std::endl;
    }
  }
 private:
  float logloss = 0.0;
};

}
#endif /* !BASE_H */
