
EXECUTABLE := maxflow-sequential

CC_FILES   := main-sequential.cpp


###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++
CXXFLAGS=
HOSTNAME=$(shell hostname)


OBJS=$(OBJDIR)/sequential.o $(OBJDIR)/benchmark.o $(OBJDIR)/main-sequential.o


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
