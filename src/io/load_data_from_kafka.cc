#include "/home/worker/xiaoshu/Field-aware-Factorization-Machine-ps/src/io/load_data_from_kafka.h"

namespace dml{
    void LoadData_from_kafka::load_data_from_kafka(int num){
        fea_matrix.clear();
	//for(int i = 0; i < num; ++i){
	yidian::data::rawlog::DumpFeature dumpFeature;
        while (g_run){
            bool isValid = ConsumeMsg(dumpFeature);
            if (isValid){
                std::cout << dumpFeature.user_id() << std::endl;
                for(yidian::data::rawlog::DocFeature docFeature : dumpFeature.docs()){
                    std::cout << docFeature.y() << std::endl;
		    label.push_back(docFeature.y());
                    for(int fid : docFeature.ids()){
                        std::cout << fid << " ";
		 	sample.push_back(h(fid));
                    }
                    std::cout << std::endl;
                }
            }
            fea_matrix.push_back(sample);
        }
        StopKafka();
	}
    }
}
