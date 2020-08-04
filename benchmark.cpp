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

void runTests(Flow *func(Graph *g, int s, int t), int numGraphs, int trials, std::string algorithmName)
{
  int refFlow;
  Flow *result;
  bool check;
  double start, finalTime;
  double times[numGraphs];

  srand(0);
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

  for (int i = 0; i < totalGraphs; i++)
  {
    printf("graph %d, %d vxs, %d edges\n", i, numVxs[i], numEdges[i]);

    for (int j = 0; j < trials; j++)
    {
      refFlow = -1;

      start = CycleTimer::currentSeconds();
      result = func(graphs[i], 0, (graphs[i]->n) - 1);
      finalTime = CycleTimer::currentSeconds() - start;
      if (i < numGraphs)
      {
        times[i] += finalTime;
      }
      if (j == (trials - 1))
      {
        times[i] /= trials;
      }
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

      printf("%s time %f\n", algorithmName.c_str(), finalTime);
    }
    free(graphs[i]->capacities);
    free(graphs[i]);
  }
}
