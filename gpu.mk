
EXECUTABLE := maxflow-parallel

CC_FILES   := main-gpu.cpp


###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++
CXXFLAGS=
HOSTNAME=$(shell hostname)


OBJS=$(OBJDIR)/gpu.o $(OBJDIR)/benchmark.o $(OBJDIR)/main-gpu.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@