/*
 * base.h
 * Copyright (C) 2018 wangxiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */
#include <string>
#include "src/io/load_data_from_disk.h"

namespace xflow {
void LoadData::load_all_data() {
  kv keyval;
  std::vector<kv> sample;
  m_data.fea_matrix.clear();
  while (!fin_.eof()) {
    std::getline(fin_, line);
    sample.clear();
    const char *pline = line.c_str();
    if (sscanf(pline, "%f%n", &y, &nchar) >= 1) {
      pline += nchar;
      m_data.label.push_back(y);
      while (sscanf(pline, "%d:%ld:%d%n", &fgid, &fid, &val, &nchar) >= 3) {
        pline += nchar;
        keyval.fgid = fgid;
        keyval.fid = fid;
        keyval.val = val;
        sample.push_back(keyval);
      }
    }
    m_data.fea_matrix.push_back(sample);
  }
  std::cout << "size : " << m_data.fea_matrix.size() << std::endl;
}

void LoadData::load_minibatch_data(int num) {
  kv keyval;
  std::vector<kv> sample;
  m_data.fea_matrix.clear();
  for (int i = 0; i < num; ++i) {
    std::getline(fin_, line);
    if (fin_.eof()) break;
    sample.clear();
    const char *pline = line.c_str();
    if (sscanf(pline, "%f%n", &y, &nchar) >= 1) {
      pline += nchar;
      m_data.label.push_back(y);
      while (sscanf(pline, "%d:%ld:%d%n", &fgid, &fid, &val, &nchar) >= 3) {
        pline += nchar;
        keyval.fgid = fgid;
        keyval.fid = fid;
        keyval.val = val;
        sample.push_back(keyval);
      }
    }
    m_data.fea_matrix.push_back(sample);
  }
}

void LoadData::load_all_hash_data() {
  kv keyval;
  std::vector<kv> sample;
  m_data.fea_matrix.clear();
  while (!fin_.eof()) {
    std::getline(fin_, line);
    sample.clear();
    const char *pline = line.c_str();
    if (sscanf(pline, "%f%n", &y, &nchar) >= 1) {
      pline += nchar;
      m_data.label.push_back(y);
      while (sscanf(pline, "%s", fid_str) >= 1) {
        pline += nchar;
        keyval.fid = h(fid_str);
        sample.push_back(keyval);
      }
    }
    m_data.fea_matrix.push_back(sample);
  }
  std::cout << "size : " << m_data.fea_matrix.size() << std::endl;
}

void LoadData::load_mibibatch_hash_data(int num) {
  kv keyval;
  std::vector<kv> sample;
  m_data.fea_matrix.clear();
  for (int i = 0; i < num; ++i) {
    std::getline(fin_, line);
    if (fin_.eof()) break;
    sample.clear();
    const char *pline = line.c_str();
    if (sscanf(pline, "%f%n", &y, &nchar) >= 1) {
      pline += nchar;
      m_data.label.push_back(y);
      while (sscanf(pline, "%s", fid_str) >= 1) {
        pline += nchar;
        keyval.fid = h(fid_str);
        sample.push_back(keyval);
      }
    }
    m_data.fea_matrix.push_back(sample);
  }
}

void LoadData::load_minibatch_hash_data_fread() {
  kv keyval;
  std::vector<kv> sample;
  m_data.label.clear();
  m_data.fea_matrix.clear();
  if (bmax < btop) {
    memmove(&buf[0], &buf[bmax], (btop - bmax) * sizeof(char));
  }
  btop -= bmax;
  btop += fread(&buf[btop], sizeof(char), buf.size() - 1 - btop, fp_);
  bmax = btop;
  if (btop + 1 == buf.size()) {
    while (bmax > 0 && buf[bmax-1] != EOF && buf[bmax-1] != '\n') --bmax;
    if (bmax != 0) {
      buf[bmax - 1] = '\0';
    } else {
      bmax = btop;
      buf[btop] = '\0';
    }
  } else {
    buf[bmax] = '\0';
  }

  char *p = &buf[0];
  while (*p != '\0') {
    char *q = p;
    while (*q != '\t') ++q;
    *q = '\0';
    float y_tmp = std::atof(p);
    if (y_tmp > 0.0000001) y = 1;
    else
      y = 0;
    m_data.label.push_back(y);
    ++q;
    p = q;
    sample.clear();
    while (*q != '\n') {
      while (*q != ' ' && *q != '\n' && *q != '\0') ++q;
      if (*q == '\n') {
        *q = '\0';
        char* pp = p;
        char* qq = pp;
        int field_index = 0;
        while (*qq != '\0') {
          while (*qq != ':') ++qq;
          *qq = '\0';
          if (field_index == 0) keyval.fgid = std::atof(pp);
          if (field_index == 1) {
            keyval.fid = h(std::string(pp));
            break;
          }
          ++qq;
          pp = qq;
          ++field_index;
        }
        // keyval.fid = h(std::string(p));
        sample.push_back(keyval);
        ++q;
        p = q;
        break;
      }
      if (*q == '\0') {
        char* pp = p;
        char* qq = pp;
        int field_index = 0;
        while (*qq != '\0') {
          while (*qq != ':') ++qq;
          *qq = '\0';
          if (field_index == 0) keyval.fgid = std::atof(pp);
          if (field_index == 1) {
            keyval.fid = h(std::string(pp));
            break;
          }
          ++qq;
          pp = qq;
          ++field_index;
        }
        // keyval.fid = h(std::string(p));
        sample.push_back(keyval);
        p = q;
        break;
      }
      *q = '\0';
      char* pp = p;
      char* qq = pp;
      int field_index = 0;
      while (*qq != '\0') {
        while (*qq != ':') ++qq;
        *qq = '\0';
        if (field_index == 0) keyval.fgid = std::atof(pp);
        if (field_index == 1) {
          keyval.fid = h(std::string(pp));
          break;
        }
        ++qq;
        pp = qq;
        ++field_index;
      }
      // keyval.fid = h(std::string(p));
      sample.push_back(keyval);
      ++q;
      p = q;
      // std::cout << "fgid = " << keyval.fgid << std::endl;
    }
    // return;
    m_data.fea_matrix.push_back(sample);
  }
}

}  // namespace xflow
