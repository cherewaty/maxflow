#include "benchmark.h"
#include "sequential.h"

int main()
{
  int numGraphs = 5;
  int trials = 1; // set to >1 for better averaging results

  runTests(edKarpSeq, numGraphs, trials, "Edmonds-Karp");
  runTests(dinicSeq, numGraphs, trials, "Dinic's");

  return 0;
}
