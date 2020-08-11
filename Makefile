
EXECUTABLE := maxflow

CC_FILES   := main-gpu.cpp


###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=nvcc
CXXFLAGS=
HOSTNAME=$(shell hostname)
CUDA_LINK_LIBS=-lcudart


OBJS=$(OBJDIR)/gpuEdKarp.o $(OBJDIR)/sequential.o $(OBJDIR)/benchmark.o $(OBJDIR)/main-gpu.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ maxflow maxflow-sequential

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) 

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(CXX) $< $(CXXFLAGS) -c -o $@
