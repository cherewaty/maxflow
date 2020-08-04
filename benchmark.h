#include <string>
#include "graphs.h"

Graph *generateGraph(int numVxs, int numEdges, int maxCap);

bool checkFlow(int totalFlow, int *flows, int n);

void runTests(Flow *func(Graph *g, int s, int t), int numGraphs, int trials, std::string algorithmName);
