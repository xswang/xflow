#include "iostream"
#include "ps.h"

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


struct ISGDHandle{
      public:
        ISGDHandle(){ ns_ = ps::NodeInfo::NumServers();}
        float alpha = 0.05, beta = 0.1, lambda1 = 50.0, lambda2 = 0.0;
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
	int SaveModel(const std::string& filename, int iter) {
            IterCmd cmd; cmd.set_save_model(); cmd.set_iter(iter);
            ps::Task task; task.set_cmd(cmd.cmd); task.set_msg(filename);
            //ps::Submit(task, ps::kServerGroup);
        }
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
	    ++cur_iter;
	    if(cur_iter % 10 == 0) SaveModel("tmp.txt", cur_iter);
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
        }

        virtual void ProcessRequest(ps::Message* request) { }
};//end class Server

}//end dmlc
