/*
 * random.h
 * Copyright (C) 2018 XiaoshuWang <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef RANDOM_H
#define RANDOM_H
#include <iostream>
#include <random>
#include <algorithm>
#include <cmath>
#include <atomic>
#include <time.h>

double current_realtime() {
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec + tp.tv_nsec * 1e-9;
}

std::default_random_engine& local_random_engine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;
    engine_wrapper_t() {
      static std::atomic<unsigned long> x(0);
      std::seed_seq sseq = {x++, x++, x++, (unsigned long)(current_realtime() * 1000)};
      engine.seed(sseq);
   }
  };
  static thread_local engine_wrapper_t r;
  return r.engine;
}


template<class T>
std::normal_distribution<T>& local_normal_real_distribution(T avg, T var) {
  static thread_local std::normal_distribution<T> distr(avg, var);
  return distr;
}
#endif /* !RANDOM_H */
