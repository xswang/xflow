#!/bin/bash
CPP = g++
CPP_tag = -std=c++11 -g -O3 -msse3

INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I./ps-lite/src -I./ps-lite/deps/include -I./dmlc-core/include/dmlc -I./src/sparsehash_cpu -I/home/worker/xiaoshu/hadoop/hadoop-2.7.1/include/
#INCLUDEPATH = -I/usr/local/include/ -I/usr/include -I./ps-lite/src -I./ps-lite/deps/include -I./dmlc-core/include/dmlc -I./src/sparsehash_cpu

#LIBRARYPATH = -L/usr/local/lib -L./ps-lite/deps/lib/ -L./ps-lite/build/ -L./dmlc-core/
#LIBRARY = -lglog -lprotobuf -lgflags -lzmq -lcityhash -llz4 -lps -ldmlc -lpthread

#INCLUDE = -I./ps-lite/src -I./ps-lite/deps/include -I./dmlc-core/include/dmlc -I./src/sparsehash_cpu -I/home/worker/xiaoshu/hadoop/hadoop-2.7.1/include/

LIBRARY = ./ps-lite/deps/lib/libglog.a ./ps-lite/deps/lib/libprotobuf.a ./ps-lite/deps/lib/libgflags.a ./ps-lite/deps/lib/libzmq.a ./ps-lite/deps/lib/libcityhash.a ./ps-lite/deps/lib/liblz4.a ./ps-lite/build/libps.a ./dmlc-core/libdmlc.a -lpthread

all: ffm_ps dump

ffm_ps: main.o $(LIBRARY)
	#$(CPP) $(CPP_tag) -o $@ $^ $(LIBRARYPATH) $(LIBRARY)
	$(CPP) $(CPP_tag) -o $@ $^ $(LIBRARY)
	rm main.o

main.o: src/main.cpp 
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/main.cpp

dump: dump.o $(LIBRARY)
	#$(CPP) $(CPP_tag) -o $@ $^ $(LIBRARYPATH) $(LIBRARY)
	$(CPP) $(CPP_tag) -o $@ $^ $(LIBRARY)
	rm dump.o

dump.o: src/dump.cpp 
	$(CPP) $(CPP_tag) $(INCLUDEPATH) -c src/dump.cpp

clean:
	rm -f *~ train
	rm -f *.o
