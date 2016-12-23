#include "load_data_from_kafka.h"

namespace dml{
    void LoadData_from_kafka::load_data_from_kafka(int num){
        fea_matrix.clear();
	//for(int i = 0; i < num; ++i){
	while(true){
	    sample.clear();
	    RdKafka::Message* msg = NULL;
	    bool isValid = false;
	    bool run = ConsumeMsg(msg, isValid);
	    if (!run){
                break;
	    }
	    if (isValid){
		int len = msg->len();
		char *content = static_cast<char *>(msg->payload()); // “123” shows “1233”, why??
		content[len] = (char)0;
		printf("%d:%s\n",len, content);
	    }
	    delete msg;
	    fea_matrix.push_back(sample);
	}
    }
}
