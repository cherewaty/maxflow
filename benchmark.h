#include <string>
#include "graphs.h"

Graph *generateGraph(int numVxs, int numEdges, int maxCap);

bool checkFlow(int totalFlow, int *flows, int n);

void runTests(std::string algorithmName, Flow *func(Graph *g, int s, int t), Graph *graphs[], int numGraphs, int trials);
