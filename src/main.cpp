#include "worker.h"
#include "server.h"

#include "ps.h"

namespace ps{
  App* App::Create(int argc, char *argv[]){
    NodeInfo n;
    if(n.IsWorker()){
	std::cout<<"create worker~"<<std::endl;
        return new ::dmlc::Worker(argv[1], argv[2]);
    }else if(n.IsServer()){
	std::cout<<"create server~"<<std::endl;
        return new ::dmlc::Server();
    }else if(n.IsScheduler){
	std::cout<<"create scheduler~"<<std::endl;
        return new ::dmlc::Scheduler();
    }
    return NULL;
  }

}//namespace ps

int64_t dmlc::ISGDHandle::new_w = 0;

int main(int argc,char *argv[]){  
    return ps::RunSystem(&argc, &argv);
}
