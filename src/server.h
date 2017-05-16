#include "iostream"
#include "ps.h"
#include "dmlc/io.h"
#include <time.h>

#define HADOOP_MODEL_PATH "/user/xiaoshu/ftrl/models_test"
#define MODEL_DUMP_INTERVAL (10 * 60) // Time interval(unit is second) to dump model.

namespace dmlc{

struct DataParCmd {
  DataParCmd() {}
  DataParCmd(int c) : cmd(c) {}
  void set_process() { cmd |= 1; }
  bool process() const { return cmd & 1; }
  int cmd = 0;
};

struct IterCmd : public DataParCmd {
  IterCmd() {}
  IterCmd(int c) : DataParCmd(c) {}
  void set_iter(int iter) { cmd |= (iter+1) << 16; }
  void set_load_model() { cmd |= 1<<1; }
  void set_save_model() { cmd |= 1<<2; }
  bool load_model() const { return cmd & 1<<1; }
  bool save_model() const { return cmd & 1<<2; }
  int iter() const { return (cmd >> 16)-1; }
};

class SaveLoadTools{
  public:
    static void GetFilesInHadoop(const std::string fileName, std::vector<std::string> &files){
      FILE *fp = fopen(fileName.c_str(),"r");
      if(NULL == fp){
        std::cerr<<"Fail to open 'hadoop fs -ls' result file tmp.txt"<<std::endl;
        exit (1) ;
      }
      const int MaxLineSize = 1024;
      char line[MaxLineSize];
      // fgets(line, MaxLineSize, fp); // Note: Such as "Found 158 items"!!!
      while (fgets(line, MaxLineSize, fp) != NULL){
        int len = strlen(line);
        line[len - 1] = '\0';
        for (int i = len - 1; i > 0; --i){
          if (line[i] == '/'){
            std::string file_name(line + i + 1);
            files.push_back(file_name);
            break;
          }
        }
      }
      fclose(fp);
    }
    static void ParseModelFile(const std::string filename, std::string &timestamp, ps::Range<uint64_t> &range){
      const char *p1 = strchr(filename.c_str(), '_');
      const char *p2 = strchr(p1 + 1, '_');
      const char *p3 = strchr(p2 + 1, '_');
      const char *p4 = strchr(p3 + 1, '.');
      timestamp = std::string(p1 + 1, p2);
      std::string rangeBegin = std::string(p2 + 1, p3);
      std::string rangeEnd = std::string(p3 + 1, p4);
      uint64_t rb = String_to_uint64(rangeBegin);
      uint64_t re = String_to_uint64(rangeEnd);
      ps::Range<uint64_t> r(rb, re);
      range = r;
    }
    static uint64_t String_to_uint64(std::string &s){
      std::stringstream strValue;
      strValue << s.c_str();
      uint64_t  value;
      strValue >>  value;
      return value;
    }
};


class Scheduler : public ps::App{
  public:
    Scheduler(){}
    ~Scheduler(){}

    virtual void ProcessResponse(ps::Message* response) { }
    virtual bool Run(){
      std::cout<<"Connected "<<ps::NodeInfo::NumServers()<<" servers and "<<ps::NodeInfo::NumWorkers()<<" workers"<<std::endl;
      //LoadModel(HADOOP_MODEL_PATH);
      ps::Task task;
      task.set_msg("StartRun");
      Submit(task, ps::kWorkerGroup);

      char now_time[1024];
      int iterator = 0;
      while(true){
        usleep(MODEL_DUMP_INTERVAL * 1e6);
        time_t tt = time(NULL);
        tm* t = localtime(&tt);
        snprintf(now_time, 1024, "%d%02d%02d%02d%02d%02d", t->tm_year + 1900,
            t->tm_mon + 1,
            t->tm_mday - 1,
            t->tm_hour,
            t->tm_min,
            t->tm_sec);
        std::string timestamp;
        timestamp = std::string(now_time);
        std::cout<<"Scheduler: Save model timestamp = "<<timestamp<<std::endl;
        SaveModel(timestamp, iterator);
      }
    }

    int SaveModel(const std::string& filename, int iter) {
      IterCmd cmd;
      cmd.set_save_model(); 
      cmd.set_iter(iter);
      ps::Task task; 
      task.set_cmd(cmd.cmd); 
      task.set_msg(filename);
      Submit(task, ps::kServerGroup);
    }
};//end class Scheduler

struct ISGDHandle{
  public:
    ISGDHandle(){ ns_ = ps::NodeInfo::NumServers();}
    float alpha = 0.01, beta = 0.1, lambda1 = 5.0, lambda2 = 0.0;
    inline void Start(bool push, int timestamp, int cmd, void* msg) { }//must has
    void Load(Stream* fi) { }//must has
    void Save(Stream *fo) const { }//must has
    inline void Finish(){ }//must has
    size_t cur_iter = 0;
  private:
    int ns_ = 0;
    static int64_t new_w;
};  


struct FTRLEntry{
  float w = 0;
  float z = 0;
  float sq_cum_grad = 0;
  inline void Load(Stream *fi) {
    fi->Read(&w, sizeof(float));
    fi->Read(&z, sizeof(float));
    fi->Read(&sq_cum_grad, sizeof(float)); 
  }//must has
  inline void Save(Stream *fo) const{
    fo->Write(&w, sizeof(float));
    fo->Write(&z, sizeof(float));
    fo->Write(&sq_cum_grad, sizeof(float));
  }//must has
  inline bool Empty() const { 
    return (w == 0) && (z == 0) && (sq_cum_grad == 0);
  }//must has
};

struct FTRLHandle : public ISGDHandle{
  public:
    inline void Push(ps::Key key, ps::Blob<const float> grad, FTRLEntry& val){
      float g = grad[0];
      float sqrt_n = val.sq_cum_grad;
      float sqrt_n_new = sqrt(sqrt_n * sqrt_n + g * g);
      val.z += g - (sqrt_n_new - sqrt_n) / alpha * val.w;
      val.sq_cum_grad = sqrt_n_new;
      if(abs(val.z) <= lambda1){
        val.w = 0.0;
      }
      else{
        float tmpr= 0.0;
        if(val.z >= 0) tmpr = val.z - lambda1;
        else tmpr = val.z + lambda1;
        float tmpl = -1 * ( (beta + val.sq_cum_grad)/alpha  + lambda2 );
        val.w = tmpr / tmpl;
      }
    }//end Push

    inline void Pull(ps::Key key, const FTRLEntry& val, ps::Blob<float>& send){
      send[0] = val.w;
    }//end Pull

  private:
};//end struct FTRLHandle

class Server : public ps::App{
  public:
    Server(){
      CreateServer<FTRLEntry, FTRLHandle>();
    }
    ~Server(){}

    template <typename Entry, typename Handle>
      void CreateServer(){
        Handle h;
        ps::OnlineServer<float, Entry, Handle> s(h, 1, 32);
        server_ = s.server();
      }

    void ProcessRequest(ps::Message* request) {
      ps::Range<ps::Key> my_range = ps::MyKeyRange();
      if(request->task.msg().size() == 0) return;
      IterCmd cmd(request->task.cmd());
      auto filename = ModelName(request->task.msg(), cmd.iter());
      if(cmd.save_model()){
        std::cout<<"Server Save model to local start: "<<filename<<std::endl;
        Stream* fo = Stream::Create(filename.c_str(), "w"); 
        server_->Save(fo);
        delete fo;
        std::cout<<"Server Save model to local finished: "<<filename<<std::endl;

        mtx_.lock();
        models_to_upload_.push(filename);
        mtx_.unlock();
      }
    }

    ps::KVStore* server_;
    std::queue<std::string> models_to_upload_;
    std::mutex mtx_;
  private:
    std::string ModelName(const std::string& base, int iter){
      std::string name = base;
      return "model_" + name + '_' + std::to_string(ps::MyKeyRange().begin()) + "_" + std::to_string(ps::MyKeyRange().end()) + ".dat"; 
    }
};//end class Server

}//end dmlc
