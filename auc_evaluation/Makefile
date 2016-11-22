#!/bash/bin
CPPFLAGS = -O2
INCLUDES = -I/usr/include/mpi/include

.PHONY:all
all: auc_calculate

%.o : src/%.cpp
	 mpicxx $(CPPFLAGS) -c $< -o $@ $(INCLUDES)

auc_calculate: main.o
	mpicxx $^ $(LDFLAGS) -o $@  $(INCLUDES)

.PHONY:clean
clean:
	rm -fr *.o auc_calculate
