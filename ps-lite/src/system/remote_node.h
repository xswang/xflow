#pragma once
#include "base/common.h"
#include "proto/task.pb.h"
#include "system/van.h"
#include "system/postoffice.h"
#include "filter/filter.h"

DECLARE_int32(timestamp_bits);

namespace ps {

// The presentation of a remote node used by Executor. It's not thread
// safe, do not use them directly.

// Track a request by its timestamp.
class RequestTracker {
 public:
  RequestTracker() { data_ = NULL; } // Lazy creation. To save half of memory in RemoteNode.
  ~RequestTracker() { delete []data_; }

  // Returns true if timestamp "ts" is marked as finished.
  bool IsFinished(int ts) {
    if (ts < 0)
    {
      return true;
    }
    if (NULL == data_)
    {
      return false;
    }

    int int_offset = 0, bit_offset = 0;
    LocateOffsets(ts, int_offset, bit_offset);
    return (data_[int_offset] & (((unsigned int)1) << bit_offset)) != 0;
  }

  // Mark timestamp "ts" as finished. Reset the future 1/4 area.
  void Finish(int ts) {
    CHECK_GE(ts, 0);
    //CHECK_LT(ts, 100000000);
    if (NULL == data_)
    {
      data_ = new unsigned int[1 << (FLAGS_timestamp_bits - 5)]; // 2^5==32==sizeof(unsigned int)
      memset(data_, 0, (1 << (FLAGS_timestamp_bits - 3)));
    }
    int int_offset = 0, bit_offset = 0;
    LocateOffsets(ts, int_offset, bit_offset);
    data_[int_offset] |= (((unsigned int)1) << bit_offset);
    {
      int int_logic_offset = (int_offset + (1 << (FLAGS_timestamp_bits - 5 - 2))); // 5:int32; 2:2^2==4,the future 1/4 location.
      data_[int_logic_offset & ((1 << (FLAGS_timestamp_bits - 5)) - 1)] = 0; // Reset a "32-bit-int area", to avoid missing a Finish msg.
    }
  }

 private:
  inline void LocateOffsets(int ts, int &int_offset, int &bit_offset)
  {
    CHECK_LT(ts, (1 << FLAGS_timestamp_bits));
    int_offset = (ts >> 5); // bit_id / 32
    bit_offset = (ts & 31); // bit_id % 32    
  }
 private:
  unsigned int *data_; // NULL or bits-size==(1 << FLAGS_timestamp_bits)
};

// A remote node
struct RemoteNode {
 public:
  RemoteNode() { }
  ~RemoteNode() {
    for (auto f : filters) delete f.second;
  }

  void EncodeMessage(Message* msg);
  void DecodeMessage(Message* msg);

  Node node;         // the remote node
  bool alive = true; // aliveness

  // timestamp tracker
  RequestTracker sent_req_tracker;
  RequestTracker recv_req_tracker;

  // node group info. if "node" is a node group, then "group" contains all node
  // pointer in this group. otherwise, group contains "this"
  void AddGroupNode(RemoteNode* rnode);
  void RemoveGroupNode(RemoteNode* rnode);
  std::vector<RemoteNode*> group;

  // keys[i] is the key range of group[i]
  std::vector<Range<Key>> keys;

 private:

  IFilter* FindFilterOrCreate(const Filter& conf);
  // key: filter_type
  std::unordered_map<int, IFilter*> filters;

};


} // namespace ps
