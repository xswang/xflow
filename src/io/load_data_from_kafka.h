#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <cstring>

#ifndef _MSC_VER
#include <sys/time.h>
#endif

#ifdef _MSC_VER
#include "../win32/wingetopt.h"
#include <atltime.h>
#elif _AIX
#include <unistd.h>
#else
#include <getopt.h>
#include <unistd.h>
#endif

/*
 * Typically include path in a real application would be
 * #include <librdkafka/rdkafkacpp.h>
 */
#include "../kafka_lib/rdkafkacpp.h"

#include "io.h"

namespace dml{

static bool run = true;
static bool exit_eof = false;
static int eof_cnt = 0;
static int partition_cnt = 0;
static int verbosity = 1;
static long msg_cnt = 0;
static int64_t msg_bytes = 0;
static void sigterm (int sig) {
  run = false;
}

RdKafka::KafkaConsumer *consumer = NULL;

class ExampleEventCb : public RdKafka::EventCb {
 public:
  void event_cb (RdKafka::Event &event) {
    switch (event.type()){
      case RdKafka::Event::EVENT_ERROR:
        std::cerr << "ERROR (" << RdKafka::err2str(event.err()) << "): " <<
            event.str() << std::endl;
        if (event.err() == RdKafka::ERR__ALL_BROKERS_DOWN)
          run = false;
        break;

      case RdKafka::Event::EVENT_STATS:
        std::cerr << "\"STATS\": " << event.str() << std::endl;
        break;

      case RdKafka::Event::EVENT_LOG:
        fprintf(stderr, "LOG-%i-%s: %s\n",
                event.severity(), event.fac().c_str(), event.str().c_str());
        break;

      case RdKafka::Event::EVENT_THROTTLE:
        std::cerr << "THROTTLED: " << event.throttle_time() << "ms by " <<
          event.broker_name() << " id " << (int)event.broker_id() << std::endl;
        break;

      default:
        std::cerr << "EVENT " << event.type() <<
            " (" << RdKafka::err2str(event.err()) << "): " <<
            event.str() << std::endl;
        break;
    }
  }
};

class LoadData_from_kafka{
public:
    LoadData_from_kafka() {
	std::string brokers = "10.103.35.18";
	std::string errstr;
	std::vector<std::string> topics;
	topics.push_back("indata_bi_dump_feature_nu");
	RdKafka::Conf *conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
	RdKafka::Conf *tconf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);

	conf->set("group.id", "helloworld", errstr);
	conf->set("metadata.broker.list", brokers, errstr);
	ExampleEventCb ex_event_cb;
	conf->set("event_cb", &ex_event_cb, errstr);
	conf->set("default_topic_conf", tconf, errstr);
	conf->set("consumer.timeout.ms", "20000", errstr);
	delete tconf;

	consumer = RdKafka::KafkaConsumer::create(conf, errstr);
	if (!consumer) { 
            std::cerr << "Failed to create consumer: " << errstr << std::endl;
            exit(1);
	}
	delete conf;
	std::cout << "% Created consumer " << consumer->name() << std::endl;
	RdKafka::ErrorCode err = consumer->subscribe(topics);
	if (err) {
            std::cerr << "Failed to subscribe to " << topics.size() << " topics: "
	        << RdKafka::err2str(err) << std::endl;
            exit(1);
	}
    }//end LoadData_from_kafka
    ~LoadData_from_kafka(){}

    bool ConsumeMsg(RdKafka::Message* &msg, bool &isValid){
        msg = consumer->consume(1000);
        isValid = false;
        switch (msg->err()) {
            case RdKafka::ERR__TIMED_OUT:
                break;
            case RdKafka::ERR_NO_ERROR:
                isValid = true;
                break;
            case RdKafka::ERR__PARTITION_EOF:
                break;
            case RdKafka::ERR__UNKNOWN_TOPIC:
            case RdKafka::ERR__UNKNOWN_PARTITION:
                std::cerr << "Consume failed: " << msg->errstr() << std::endl;
                run = false;
                break;
            default:
            /* Errors */
            std::cerr << "Consume failed: " << msg->errstr() << std::endl;
            run = false;
        }
        return run;
    }

    int StopKafka(){
	consumer->close();
	delete consumer;
	std::cerr << "% Consumed " << msg_cnt << " messages ("
		<< msg_bytes << " bytes)" << std::endl;
	RdKafka::wait_destroyed(5000);
	return 0;
    }

    void load_data_from_kafka(int num);

public:
    std::set<long int> feaIdx;
    std::set<long int>::iterator setIter;

    typedef kv key_val;
    std::vector<key_val> sample;
    std::vector<std::vector<key_val>> fea_matrix;
    std::vector<int> label;

};

}
