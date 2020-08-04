#include "benchmark.h"
#include "gpu.h"

int main()
{
  srand(0);
  int trials = 1; // set to >1 for better averaging results

  int numGraphs = 5;
  int numVxs[] = {500, 500, 2000, 2000, 5000, 5000, 20000, 20000, 40000, 40000};
  int numEdges[] = {1000, 5000, 5000, 10000, 10000, 20000, 50000, 100000, 100000, 500000};
  int maxCap = 50;
  Graph *graphs[numGraphs];

  for (int i = 0; i < numGraphs; i++)
  {
    graphs[i] = generateGraph(numVxs[i], numEdges[i], maxCap);
  }

  runTests("Edmonds-Karp", edKarpGpu, graphs, numGraphs, trials);
  runTests("Dinic's", dinicGpu, graphs, numGraphs, trials);

  return 0;
}
