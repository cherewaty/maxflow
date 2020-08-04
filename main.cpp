#include "benchmark.h"

int main()
{
  int numGraphs = 5;
  int trials = 1; // set to >1 for better averaging results

  runTests(numGraphs, trials);
  return 0;
}
