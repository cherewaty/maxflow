#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include "cycle_timer.h"
#include "graphs.h"

#define IDX(i, j, n) ((i) * (n) + (j))

/*
*   Signifcant portions of source from https://github.com/vulq/Flo
*/

// Generates a random directed graph that ensures there exists some s-t path, where
// s is node 0 and t is node numVxs - 1.
Graph *generateGraph(int numVxs, int numEdges, int maxCap)
{
  Graph *g = (Graph *)malloc(sizeof(Graph));
  g->n = numVxs;
  g->capacities = (int *)calloc((numVxs * numVxs), sizeof(int));
  std::vector<int> path;
  for (int i = 1; i < numVxs; i++)
  {
    path.push_back(i);
  }

  std::random_device rd;
  std::mt19937 randomizer(rd());
  std::shuffle(path.begin(), path.end(), randomizer);

  int t = numVxs - 1;
  int first = path.at(0);
  int capacity = (rand() % (maxCap - 1)) + 1;
  g->capacities[IDX(0, first, g->n)] = capacity;
  int remaining = numEdges - 1;
  int last = first;
  for (std::vector<int>::iterator it = (path.begin()++); ((it != path.end()) && (*it != t)); it++)
  {
    capacity = (rand() % (maxCap - 1)) + 1;
    g->capacities[IDX(last, (*it), g->n)] = capacity;
    last = *it;
    remaining--;
  }
  capacity = (rand() % (maxCap - 1)) + 1;
  g->capacities[IDX(last, t, g->n)] = capacity;
  remaining--;
  for (int i = 0; i < remaining; i++)
  {
    capacity = (rand() % (maxCap - 1)) + 1;
    int j = rand() % g->n;
    int k = rand() % g->n;
    if (j != k)
    {
      g->capacities[IDX(j, k, g->n)] = capacity;
    }
  }
  return g;
}

// for each non s or t node, check flow in == flow out, and check flow
// out of s equals flow into t equals total flow. Also check that flows
// are symmetric (i.e. F[u][v] == -F[v][u])
bool checkFlow(int totalFlow, int *flows, int n)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      if (flows[IDX(i, j, n)] != -flows[IDX(j, i, n)])
      {
        return false;
      }
    }
  }
  for (int i = 1; i < (n - 1); i++)
  {
    int flowIn = 0;
    int flowOut = 0;
    for (int j = 0; j < n; j++)
    {
      int edgeFlow = flows[IDX(i, j, n)];
      if (edgeFlow > 0)
      {
        flowIn += edgeFlow;
      }
      else
      {
        flowOut -= edgeFlow;
      }
    }
    if (flowIn != flowOut)
    {
      return false;
    }
  }
  int sFlow = 0;
  int tFlow = 0;
  for (int i = 0; i < n; i++)
  {
    sFlow += flows[IDX(0, i, n)];
    tFlow += flows[IDX(i, (n - 1), n)];
  }
  return (sFlow == tFlow) && (sFlow == totalFlow);
}

void runTests(std::string algorithmName, Flow *func(Graph *g, int s, int t), Graph *graphs[], int numGraphs, int trials)
{
  int refFlow;
  Flow *result;
  bool check;
  double start, finalTime;
  double times[numGraphs];

  for (int i = 0; i < numGraphs; i++)
  {
    printf("%s, graph %d\n", algorithmName.c_str(), i);

    for (int j = 0; j < trials; j++)
    {
      refFlow = -1;

      start = CycleTimer::currentSeconds();
      result = func(graphs[i], 0, (graphs[i]->n) - 1);
      finalTime = CycleTimer::currentSeconds() - start;

      printf("Trial %d - max flow computed: %d\n", (j + 1), result->maxFlow);

      times[j] = finalTime;

      check = checkFlow(result->maxFlow, result->finalEdgeFlows, graphs[i]->n);

      if (!check)
      {
        std::cout << algorithmName.c_str() << " flows don't agree with max flow on graph " << i << std::endl;
      }

      if ((refFlow != -1) && (result->maxFlow != refFlow))
      {
        std::cout << algorithmName.c_str() << " flow doesn't agree with refFlow on graph " << i << std::endl;
      }

      if (refFlow == -1)
      {
        refFlow = result->maxFlow;
      }

      free(result->finalEdgeFlows);
      free(result);
    }

    // calculate the average
    double sum;
    for (int k = 0; k < trials; k++)
    {
      sum += times[k];
    }

    printf("Average seconds to compute %f\n\n", (sum / trials));
  }
}
