#include <string>
#include "graphs.h"

Graph *generateGraph(int numVxs, int numEdges, int maxCap);

bool checkFlow(int totalFlow, int *flows, int n);

void runTests(Graph *graphs[], Flow *func(Graph *g, int s, int t), int numGraphs, int totalGraphs, int trials, std::string algorithmName);
