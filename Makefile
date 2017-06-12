#!/bin/bash
CPP = g++
CPP_tag = -std=c++11 -g -O3 -msse3

#INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I./ps-lite/src -I./ps-lite/deps/include -I./dmlc-core/include/dmlc -I./thirdparty/librdkafka-master/src-cpp
INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I./ps-lite/src -I./ps-lite/deps/include -I./dmlc-core/include/dmlc 

#LIBRARY = /usr/local/lib/librdkafka++.a /usr/local/lib/librdkafka.a ./ps-lite/deps/lib/libglog.a ./ps-lite/deps/lib/libprotobuf.a ./ps-lite/deps/lib/libgflags.a ./ps-lite/deps/lib/libzmq.a ./ps-lite/deps/lib/libcityhash.a ./ps-lite/deps/lib/liblz4.a ./ps-lite/build/libps.a ./dmlc-core/libdmlc.a -lz -lpthread -lssl -lcrypto
LIBRARY = ./ps-lite/deps/lib/libglog.a ./ps-lite/deps/lib/libprotobuf.a ./ps-lite/deps/lib/libgflags.a ./ps-lite/deps/lib/libzmq.a ./ps-lite/deps/lib/libcityhash.a ./ps-lite/deps/lib/liblz4.a ./ps-lite/build/libps.a ./dmlc-core/libdmlc.a -lz -lpthread

all: ffm_ps dump


ffm_ps: main.o service_dump_feature.pb.o load_data_from_local.o $(LIBRARY)
	$(CPP) $(CPP_tag) -o $@ $^ $(LIBRARY)
	rm main.o
	rm service_dump_feature.pb.o
	mkdir bin
	mv ffm_ps bin

main.o: src/main.cpp 
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp

service_dump_feature.pb.o: src/io/service_dump_feature.pb.cc
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/io/service_dump_feature.pb.cc

load_data_from_local.o: src/io/load_data_from_local.cc
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/io/load_data_from_local.cc

dump: dump.o $(LIBRARY)
	$(CPP) $(CPP_tag) -o $@ $^ $(LIBRARY)
	rm dump.o
	mv dump bin

dump.o: src/dump.cpp 
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/dump.cpp

clean:
	rm -f *~ train
	rm -f *.o
	rm -rf bin
