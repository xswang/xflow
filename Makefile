#!/bin/bash
CPP = g++
CPP_tag = -std=c++11 -g -O3 -msse3

#INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I./ps-lite/src -I./ps-lite/deps/include -I./dmlc-core/include/dmlc  -I./ps-lite/include/ps -I./ps-lite/include -I./dmlc-core/include
INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I./ps-lite/src -I./ps-lite/deps/include -I./ps-lite/include/ps -I./ps-lite/include

LIBRARY = ./ps-lite/deps/lib/libprotobuf.a ./ps-lite/deps/lib/libzmq.a ./ps-lite/build/libps.a

all: lr_ps


lr_ps: main.o service_dump_feature.pb.o load_data_from_local.o $(LIBRARY)
	$(CPP) $(CPP_tag) -o $@ $^ $(LIBRARY)
	rm main.o
	rm service_dump_feature.pb.o
	rm load_data_from_local.o
	rm -rf bin
	mkdir bin
	mv lr_ps bin

main.o: src/main.cpp 
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp

service_dump_feature.pb.o: src/io/service_dump_feature.pb.cc
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/io/service_dump_feature.pb.cc

load_data_from_local.o: src/io/load_data_from_local.cc
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/io/load_data_from_local.cc

clean:
	rm -f *~ train
	rm -f *.o
	rm -rf bin
