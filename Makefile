#!/bin/bash
CPP = g++
CPP_tag = -std=c++11
DEPS_PATH = ./ps-lite/deps 
INCLUDE = -I./ps-lite/src -I./ps-lite/deps/include -I./dmlc-core/include/dmlc

#LDFLAGS = $(addprefix $(DEPS_PATH)/lib/, libglog.a libprotobuf.a libgflags.a libzmq.a libcityhash.a liblz4.a)
LDFLAGS = ./ps-lite/build/libps.a ./dmlc-core/libdmlc.a ./ps-lite/deps/lib/libglog.a ./ps-lite/deps/lib/libprotobuf.a ./ps-lite/deps/lib/libgflags.a ./ps-lite/deps/lib/libzmq.a ./ps-lite/deps/lib/libcityhash.a ./ps-lite/deps/lib/liblz4.a -lpthread

all: ffm_ps

ffm_ps: main.o $(LDFLAGS)
	$(CPP) $(CPP_tag) -o $@ $^ $(INCLUDE)
	rm main.o

main.o: src/main.cpp
	$(CPP) $(CPP_tag) -c src/main.cpp $(INCLUDE)

#dump_model: dump.o ./repo/dmlc-core/libdmlc.a
#	$(CPP) $(CPP_tag) -o $@ $^ $(INCLUDE)

#dump.o: src/dump.cc ./repo/dmlc-core/libdmlc.a ./repo/ps-lite/deps/lib/libglog.a 
#	$(CPP) $(CPP_tag) -c src/dump.cc $(INCLUDE)
	

clean:
	rm -f *~ train
	rm -f *.o
