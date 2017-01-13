#include "rdkafka_example_gt.h"


int main(int argc, char **argv) 
{
    yidian::data::rawlog::DumpFeature dumpFeature;

    StartKafka();
    while (g_run)
    {
        bool isValid = ConsumeMsg(dumpFeature);
        if (isValid)
        {
            std::cout << dumpFeature.user_id() << std::endl;
            for(yidian::data::rawlog::DocFeature docFeature : dumpFeature.docs()){
                std::cout << docFeature.y() << std::endl;
                for(int fid : docFeature.ids()){
                    std::cout << fid << " ";
                }
                std::cout << std::endl;
            }
        }
    }
    StopKafka();
    return 0;
}
