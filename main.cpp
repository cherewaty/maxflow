#include "benchmark.h"
#include "sequential.h"

int main()
{
  srand(0);

  int numGraphs = 5;
  int trials = 1; // set to >1 for better averaging results

  int smallGraphNum = 0; // used for correctness testing, not benchmarking
  int totalGraphs = numGraphs + smallGraphNum;
  int numVxs[] = {500, 500, 2000, 2000, 5000, 5000, 20000, 20000, 40000, 40000};
  int numEdges[] = {1000, 5000, 5000, 10000, 10000, 20000, 50000, 100000, 100000, 500000};
  int maxCap = 50;
  Graph *graphs[totalGraphs];

  for (int i = 0; i < numGraphs; i++)
  {
    graphs[i] = generateGraph(numVxs[i], numEdges[i], maxCap);
  }

  // generate small graphs too
  for (int i = numGraphs; i < totalGraphs; i++)
  {
    int vxs = (rand() % 1000) + 100;
    int maxEdges = (vxs * (vxs - 1)) / 2;
    int edges = (rand() % 20000) + vxs;
    edges = std::min(edges, maxEdges);
    int cap = (rand() % 1000) + 20;
    graphs[i] = generateGraph(vxs, edges, cap);
  }

  runTests(graphs, edKarpSeq, numGraphs, totalGraphs, trials, "Edmonds-Karp");
  runTests(graphs, dinicSeq, numGraphs, totalGraphs, trials, "Dinic's");

  return 0;
}
