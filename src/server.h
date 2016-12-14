#include "iostream"
#include "ps.h"
#include "utils.h"

namespace dmlc{
struct ISGDHandle{
      public:
        ISGDHandle(){ ns_ = ps::NodeInfo::NumServers();}
        float alpha = 1.0, beta = 1.0, lambda1 = 0.001, lambda2 = 0.0;
        inline void Start(bool push, int timestamp, int cmd, void* msg) { }//must has
        void Load(Stream* fi) { }//must has
        void Save(Stream *fo) const { }//must has
        inline void Finish(){ }//must has
      private:
        int ns_ = 0;
        static int64_t new_w;
};  

template <typename T> 
inline void TSave(Stream* fo, T* const ptr){
    fo->Write(&ptr->w, sizeof(float));
}

struct FTRLEntry{
    float w = 0.99;
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
        }

        virtual void ProcessRequest(ps::Message* request) { }
};//end class Server

}//end dmlc
