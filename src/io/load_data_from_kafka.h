#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <sys/time.h>

#ifdef _AIX
#include <unistd.h>
#else
#include <getopt.h>
#include <unistd.h>
#endif

/*
 * Typically include path in a real application would be
 * #include <librdkafka/rdkafkacpp.h>
 */
#include <set>
#include "rdkafkacpp.h"
#include "service_dump_feature.pb.h"
#include "io.h"

#define KAFKA_BROKERS "10.120.14.11:9092,10.120.14.12:9092"
#define KAFKA_TOPIC   "indata_bi_dump_feature_nu"

namespace dml{

// Kafka statistics.
static int eof_cnt = 0;
static int partition_cnt = 0;
static long msg_cnt = 0;
static int64_t msg_bytes = 0;

bool g_isRestartKafka = false;

/**
 * @brief format a string timestamp from the current time
 */
static void print_time () 
{
  struct timeval tv;
  char buf[64];
  gettimeofday(&tv, NULL);
  strftime(buf, sizeof(buf) - 1, "%Y-%m-%d %H:%M:%S", localtime(&tv.tv_sec));
  fprintf(stderr, "%s.%03d: ", buf, (int)(tv.tv_usec / 1000));
}

class ExampleEventCb : public RdKafka::EventCb {
 public:
  void event_cb (RdKafka::Event &event) {

    std::cerr<<std::endl;
    print_time();

    switch (event.type())
    {
      case RdKafka::Event::EVENT_ERROR:
        std::cerr << "ERROR (" << RdKafka::err2str(event.err()) << "): " <<
            event.str() << std::endl;
        g_isRestartKafka = true;
        if (event.err() == RdKafka::ERR__ALL_BROKERS_DOWN)
          std::cerr << "ERROR: RdKafka::ERR__ALL_BROKERS_DOWN" << std::endl;
          exit(1);
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


class ExampleRebalanceCb : public RdKafka::RebalanceCb {
private:
  static void part_list_print (const std::vector<RdKafka::TopicPartition*>&partitions)
  {
    for (unsigned int i = 0 ; i < partitions.size() ; i++)
      std::cerr << partitions[i]->topic() <<
        "[" << partitions[i]->partition() << "]"
        << "(" << partitions[i]->offset() << ")\n";
    std::cerr << "\n";
  }

public:
  void rebalance_cb (RdKafka::KafkaConsumer *consumer,
                      RdKafka::ErrorCode err,
                      std::vector<RdKafka::TopicPartition*> &partitions) 
  {
    std::cout << "RebalanceCb: " << RdKafka::err2str(err) << ": " << std::endl;

    part_list_print(partitions);

    if (err == RdKafka::ERR__ASSIGN_PARTITIONS) 
    {
      std::cout << "ERR__ASSIGN_PARTITIONS" << std::endl;
      for (int i = 0; i < partitions.size(); ++i)
      {
        std::cout << "ERR__ASSIGN_PARTITIONS " << i << " OFFSET_BEGINNING" << std::endl;
        partitions[i]->set_offset(RdKafka::Topic::OFFSET_BEGINNING); // Use this in offline.
        //partitions[i]->set_offset(RdKafka::Topic::OFFSET_STORED); // TODO: Use this in online.
        //partitions[i]->set_offset(RdKafka::Topic::OFFSET_END); // Just for test.
      }
      consumer->assign(partitions);
      partition_cnt = (int)partitions.size();
    } else 
    {
      std::cout << "consumer->unassign()" << std::endl;
      consumer->unassign();
      partition_cnt = 0;
    }
    eof_cnt = 0;
  }
};


RdKafka::KafkaConsumer *consumer = NULL;
ExampleRebalanceCb ex_rebalance_cb; // Note: MUST no destructed during consuming, or else segment error!!!
ExampleEventCb ex_event_cb;         // Note: MUST no destructed during consuming, or else segment error!!!
std::vector<std::string> topics;


class LoadData_from_kafka
{
public:
  LoadData_from_kafka()
  {
    StartKafka(KAFKA_BROKERS, KAFKA_TOPIC);
  }

  void StartKafka(const std::string &brokers, const std::string &topic)
  {
    std::cout << "Broker=" << brokers.c_str() << std::endl;
    std::string errstr;

    RdKafka::Conf *conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    RdKafka::Conf *tconf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);

    conf->set("rebalance_cb", &ex_rebalance_cb, errstr);
    if (conf->set("group.id", "helloworld", errstr) != RdKafka::Conf::CONF_OK)
    {
      std::cerr << errstr << std::endl;
      exit(1);
    }
    topics.push_back(topic);
    conf->set("metadata.broker.list", brokers, errstr);
    conf->set("event_cb", &ex_event_cb, errstr);
    conf->set("default_topic_conf", tconf, errstr);
    conf->set("log.connection.close", "false", errstr);
    delete tconf;

    consumer = RdKafka::KafkaConsumer::create(conf, errstr);
    if (!consumer) 
    {
      std::cerr << "Failed to create consumer: " << errstr << std::endl;
      exit(1);
    }
    delete conf;
    std::cout << "Created consumer " << consumer->name() << std::endl;

    RdKafka::ErrorCode err = consumer->subscribe(topics);
    if (err) 
    {
      std::cerr << "Failed to subscribe to " << topics.size() << " topics: "
                << RdKafka::err2str(err) << std::endl;
      exit(1);
    }
  }

  void ConsumeMsg(RdKafka::Message* &message, bool &isValid, bool &isRun) 
  {
    if (g_isRestartKafka)
    {
      std::cout << "Stop Kafka" << std::endl;
      StopKafka();
      std::cout << "Restart Kafka" << std::endl;
      StartKafka(KAFKA_BROKERS, KAFKA_TOPIC);
      g_isRestartKafka = false;
    }
    message = consumer->consume(1000);
    switch (message->err()) 
    {
      case RdKafka::ERR__TIMED_OUT:
        isValid = false;
        isRun = true;
        break;

      case RdKafka::ERR_NO_ERROR:
        /* Real message */
        msg_cnt++;
        msg_bytes += message->len();
        //std::cout << "Read msg in partition" << message->partition() << " at offset " << message->offset() << std::endl;
        isValid = true;
        isRun = true;
        break;

      case RdKafka::ERR__PARTITION_EOF:
        /* Last message */
        /*
        if (++eof_cnt == partition_cnt) 
        {
          std::cerr << "%% EOF reached for all " << partition_cnt <<
              " partition(s)" << std::endl;
        }
        */
        std::cout << "ERR__PARTITION_EOF" << std::endl;
        isValid = false;
        isRun = true;
        break;

      case RdKafka::ERR__UNKNOWN_TOPIC:
      case RdKafka::ERR__UNKNOWN_PARTITION:
        std::cerr << "Consume failed: " << message->errstr() << std::endl;
        isValid = false;
        isRun = false;
        break;

      default:
        /* Errors */
        std::cerr << "Consume failed: " << message->errstr() << std::endl;
        isValid = false;
        isRun = false;
    }
  }

  void StopKafka()
  {
    consumer->close();
    delete consumer;
    std::cerr << "Consumed " << msg_cnt << " messages ("
              << msg_bytes << " bytes)" << std::endl;
    RdKafka::wait_destroyed(5000);
  }

  void load_data_from_kafka()
  { 
    kv keyval;
    m_data.fea_matrix.clear();    
    m_data.label.clear();
    std::vector<kv> sample;
int label1 = 0;    
    while(true){
      RdKafka::Message* msg = NULL;
      bool isValid = false;
      bool isRun = false;
      ConsumeMsg(msg, isValid, isRun);
      if (!isRun){
        break;
      }
      if (isValid){
        yidian::data::rawlog::DumpFeature dumpFeature;
        dumpFeature.ParseFromArray(msg->payload(), (int) msg->len());
        //std::cout << dumpFeature.user_id() << std::endl;
        for(yidian::data::rawlog::DocFeature docFeature : dumpFeature.docs()){
          //std::cout << docFeature.y() << std::endl;
          int lb = 0;
          if (docFeature.y() > 0)
          {
            lb = 1;
            ++label1;
          }
          m_data.label.push_back(lb);
          sample.clear();
          for(int fid : docFeature.ids()){ // TODO: Is it "int"???
            //std::cout << fid << " ";
            keyval.fid = h(fid);
            sample.push_back(keyval);
          }
          //std::cout << std::endl;
          m_data.fea_matrix.push_back(sample);
        }
            static int batch_id = 0;
            if (m_data.fea_matrix.size() >= 10000 * 10)
            {
              ++batch_id;
              std::cout << "Batch_id = " << batch_id << std::endl;
              break;
            }
      }
      delete msg;
    }
    std::cout << "Label1 = " << label1 << "; Total = " << m_data.fea_matrix.size() << std::endl;
    std::cout << "load_data_from_kafka finished" << std::endl;       
  }

private:
  std::hash<int> h;

public:
    Data m_data; 
};//end LoadData_from_kafka class;

}//end namespace dml
