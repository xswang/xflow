#include "iostream"
#include "ps.h"
#include "dmlc/io.h"
#include <time.h>

namespace dmlc{

struct DataParCmd {
  DataParCmd() {}
  DataParCmd(int c) : cmd(c) {}

  void set_process() { cmd |= 1; }
      // accessors
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

class Scheduler : public ps::App{
    public:
        Scheduler(){}
        ~Scheduler(){}

	    virtual void ProcessResponse(ps::Message* response) { }
	    virtual bool Run(){
	        std::cout<<"Connected "<<ps::NodeInfo::NumServers()<<" servers and "<<ps::NodeInfo::NumWorkers()<<" workers"<<std::endl;

            time_t tt = time(NULL);
            tm* t = localtime(&tt);
            char now_time[1024];
            int iterator = 0;
            while(1){
                usleep(300 * 1e6);
                snprintf(now_time, 1024, "%d-%02d-%02d %02d:%02d:%02d", t->tm_year + 1900,
                                                                        t->tm_mon + 1,
                                                                        t->tm_mday,
                                                                        t->tm_hour,
                                                                        t->tm_min,
                                                                        t->tm_sec);
                std::string timestamp;
                timestamp = std::string(now_time);
                if(iterator % 5 == 0)SaveModel("model", iterator);
                iterator++;
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

template <typename T> 
inline void TSave(Stream* fo, T* const ptr){
    fo->Write(&ptr->w, sizeof(float));
}

struct FTRLEntry{
    float w = 0.0;
    float z = 0.0;
    float sq_cum_grad = 0.0;
    inline void Load(Stream *fi) { }//must has
    inline void Save(Stream *fo) const {
	TSave(fo, this);
    }//must has
    inline bool Empty() const { }//must has
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
            std::cout<<"server ProcessRequest "<<std::endl;
            if(request->task.msg().size() == 0) return;
            IterCmd cmd(request->task.cmd());
            auto filename = ModelName(request->task.msg(), cmd.iter());
            if(cmd.save_model()){
                Stream* fo = Stream::Create(filename.c_str(), "w"); 
                server_->Save(fo);
                delete fo;
            }else if(cmd.load_model()){
                Stream* fi = Stream::Create(filename.c_str(), "r");
                server_->Load(fi);
                delete fi;
            }
        }
        ps::KVStore* server_;
    private:
        std::string ModelName(const std::string& base, int iter){
            std::string name = base;
            if(iter >= 0) name += "_iter_" + std::to_string(iter);
            return name + "_part_" + std::to_string(ps::NodeInfo::MyRank());
        }
};//end class Server

}//end dmlc
