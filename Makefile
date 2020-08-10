
EXECUTABLE := maxflow

CC_FILES   := main-gpu.cpp


###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++
NVCC=nvcc
CXXFLAGS=
HOSTNAME=$(shell hostname)
CUDA_LINK_LIBS=-lcuda


OBJS=$(OBJDIR)/gpuEdKarp.o $(OBJDIR)/benchmark.o $(OBJDIR)/main-gpu.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ maxflow maxflow-sequential

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(CUDA_LINK_LIBS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(CXXFLAGS) -c -o $@
