/*
 * librdkafka - Apache Kafka C library
 *
 * Copyright (c) 2014, Magnus Edenhill
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: 
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Apache Kafka consumer & producer example programs
 * using the Kafka driver from librdkafka
 * (https://github.com/edenhill/librdkafka)
 */

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <csignal>
#include <cstring>

#include <getopt.h>

/*
 * Typically include path in a real application would be
 * #include <librdkafka/rdkafkacpp.h>
 */
#include <librdkafka/rdkafkacpp.h>

#include "service_dump_feature.pb.h"

namespace dml{

static void metadata_print(const std::string &topic,
                           const RdKafka::Metadata *metadata) {
    std::cout << "Metadata for " << (topic.empty() ? "" : "all topics")
              << "(from broker " << metadata->orig_broker_id()
              << ":" << metadata->orig_broker_name() << std::endl;

    /* Iterate brokers */
    std::cout << " " << metadata->brokers()->size() << " brokers:" << std::endl;
    RdKafka::Metadata::BrokerMetadataIterator ib;
    for (ib = metadata->brokers()->begin();
         ib != metadata->brokers()->end();
         ++ib) {
        std::cout << "  broker " << (*ib)->id() << " at "
                  << (*ib)->host() << ":" << (*ib)->port() << std::endl;
    }
    /* Iterate topics */
    std::cout << metadata->topics()->size() << " topics:" << std::endl;
    RdKafka::Metadata::TopicMetadataIterator it;
    for (it = metadata->topics()->begin();
         it != metadata->topics()->end();
         ++it) {
        std::cout << "  topic " << *(*it)->topic() << " with "
                  << (*it)->partitions()->size() << " partitions" << std::endl;

        if ((*it)->err() != RdKafka::ERR_NO_ERROR) {
            std::cout << " " << err2str((*it)->err());
            if ((*it)->err() == RdKafka::ERR_LEADER_NOT_AVAILABLE)
                std::cout << " (try again)";
        }
        std::cout << std::endl;

        /* Iterate topic's partitions */
        RdKafka::TopicMetadata::PartitionMetadataIterator ip;
        for (ip = (*it)->partitions()->begin();
             ip != (*it)->partitions()->end();
             ++ip) {
            std::cout << "    partition " << (*ip)->id()
                      << " leader " << (*ip)->leader()
                      << ", replicas: ";

            /* Iterate partition's replicas */
            RdKafka::PartitionMetadata::ReplicasIterator ir;
            for (ir = (*ip)->replicas()->begin();
                 ir != (*ip)->replicas()->end();
                 ++ir) {
                std::cout << (ir == (*ip)->replicas()->begin() ? "," : "") << *ir;
            }

            /* Iterate partition's ISRs */
            std::cout << ", isrs: ";
            RdKafka::PartitionMetadata::ISRSIterator iis;
            for (iis = (*ip)->isrs()->begin(); iis != (*ip)->isrs()->end(); ++iis)
                std::cout << (iis == (*ip)->isrs()->begin() ? "," : "") << *iis;

            if ((*ip)->err() != RdKafka::ERR_NO_ERROR)
                std::cout << ", " << RdKafka::err2str((*ip)->err()) << std::endl;
            else
                std::cout << std::endl;
        }
    }
}

static bool g_run = true;
static bool exit_eof = false;

static void sigterm(int sig) {
    g_run = false;
}


class ExampleDeliveryReportCb : public RdKafka::DeliveryReportCb {
public:
    void dr_cb(RdKafka::Message &message) {
        std::cout << "Message delivery for (" << message.len() << " bytes): " <<
                  message.errstr() << std::endl;
    }
};


class ExampleEventCb : public RdKafka::EventCb {
public:
    void event_cb(RdKafka::Event &event) {
        switch (event.type()) {
            case RdKafka::Event::EVENT_ERROR:
                std::cerr << "ERROR (" << RdKafka::err2str(event.err()) << "): " <<
                          event.str() << std::endl;
                if (event.err() == RdKafka::ERR__ALL_BROKERS_DOWN)
                    g_run = false;
                break;

            case RdKafka::Event::EVENT_STATS:
                std::cerr << "\"STATS\": " << event.str() << std::endl;
                break;

            case RdKafka::Event::EVENT_LOG:
                fprintf(stderr, "LOG-%i-%s: %s\n",
                        event.severity(), event.fac().c_str(), event.str().c_str());
                break;

            default:
                std::cerr << "EVENT " << event.type() <<
                          " (" << RdKafka::err2str(event.err()) << "): " <<
                          event.str() << std::endl;
                break;
        }
    }
};


/* Use of this partitioner is pretty pointless since no key is provided
 * in the produce() call. */
class MyHashPartitionerCb : public RdKafka::PartitionerCb {
public:
    int32_t partitioner_cb(const RdKafka::Topic *topic, const std::string *key,
                           int32_t partition_cnt, void *msg_opaque) {
        return djb_hash(key->c_str(), key->size()) % partition_cnt;
    }

private:

    static inline unsigned int djb_hash(const char *str, size_t len) {
        unsigned int hash = 5381;
        for (size_t i = 0; i < len; i++)
            hash = ((hash << 5) + hash) + str[i];
        return hash;
    }
};

bool msg_consume(RdKafka::Message *message, void *opaque, yidian::data::rawlog::DumpFeature &dumpFeature) {
    bool isValid = false;
    switch (message->err()) {
        case RdKafka::ERR__TIMED_OUT:
            break;

        case RdKafka::ERR_NO_ERROR:
            /* Real message */
//            std::cout << "Read msg at offset " << message->offset() << std::endl;
//            if (message->key()) {
//                std::cout << "Key: " << *message->key() << std::endl;
//            }
            dumpFeature.ParseFromArray(message->payload(), (int) message->len());
            isValid = true;
            break;

        case RdKafka::ERR__PARTITION_EOF:
            /* Last message */
            if (exit_eof) {
                g_run = false;
            }
            break;

        case RdKafka::ERR__UNKNOWN_TOPIC:
        case RdKafka::ERR__UNKNOWN_PARTITION:
            std::cerr << "Consume failed: " << message->errstr() << std::endl;
            g_run = false;
            break;

        default:
            /* Errors */
            std::cerr << "Consume failed: " << message->errstr() << std::endl;
            g_run = false;
    }
    return isValid;
}



RdKafka::Consumer *consumer = NULL;
RdKafka::Topic *topic = NULL;
int32_t partition = 0; // ????????????????????????????????????? How to consume multiple partitions?
class LoadData_from_kafka{
public:
  LoadData_from_kafka() {
    std::string brokers = "10.103.35.11:9092";
    std::string errstr;
    std::string topic_str = "indata_bi_dump_feature_video_base";
    int64_t start_offset = RdKafka::Topic::OFFSET_BEGINNING;
    MyHashPartitionerCb hash_partitioner;

    RdKafka::Conf *conf = RdKafka::Conf::create(RdKafka::Conf::CONF_GLOBAL);
    RdKafka::Conf *tconf = RdKafka::Conf::create(RdKafka::Conf::CONF_TOPIC);
    conf->set("metadata.broker.list", brokers, errstr);
    ExampleEventCb ex_event_cb;
    conf->set("event_cb", &ex_event_cb, errstr);

    {
        if (topic_str.empty()) {
            std::cerr << "Invalid topic!" << std::endl;
            exit(1);
        }
        consumer = RdKafka::Consumer::create(conf, errstr);
        if (!consumer) {
            std::cerr << "Failed to create consumer: " << errstr << std::endl;
            exit(1);
        }
        std::cout << "% Created consumer " << consumer->name() << std::endl;
        topic = RdKafka::Topic::create(consumer, topic_str,
        if (!topic) {
            std::cerr << "Failed to create topic: " << errstr << std::endl;
            exit(1);
        }
        RdKafka::ErrorCode resp = consumer->start(topic, partition, start_offset);
        if (resp != RdKafka::ERR_NO_ERROR) {
            std::cerr << "Failed to start consumer: " <<
                      RdKafka::err2str(resp) << std::endl;
            exit(1);
        }
    }
    return 0;
  }
  ~LoadData_from_kafka(){}

bool ConsumeMsg(yidian::data::rawlog::DumpFeature &dumpFeature){
    RdKafka::Message *msg = consumer->consume(topic, partition, 1000);
    bool isValid = msg_consume(msg, NULL, dumpFeature);
    delete msg;
    consumer->poll(0);
    return isValid;
}

int StopKafka(){
        consumer->stop(topic, partition);
        consumer->poll(1000);
        delete topic;
        delete consumer;
    RdKafka::wait_destroyed(5000);
    return 0;
}
};

}
