#include <algorithm>
#include <limits>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sequential.h"

#define IDX(i, j, n) ((i) * (n) + (j))

/*
*   Source from https://github.com/vulq/Flo
*/

int sequentialBFS(Graph *g, int *flowMatrix, int *parents, int *pathCapacities, int s, int t)
{
  memset(parents, -1, (g->n * sizeof(int)));
  memset(pathCapacities, 0, (g->n * sizeof(int)));
  parents[s] = s;
  pathCapacities[s] = std::numeric_limits<int>::max();
  std::queue<int> bfsQueue;
  bfsQueue.push(s);
  while (!bfsQueue.empty())
  {
    int u = bfsQueue.front();
    bfsQueue.pop();
    for (int v = 0; v < g->n; v++)
    {
      if (u == v)
        continue;
      printf("Loop U %d , V %d\n",u,v);
      printf("flowmatrix bfs %d\n",flowMatrix[IDX(u, v, g->n)]);
      printf("cap bfs %d\n",g->capacities[IDX(u, v, g->n)]);
      int residual = g->capacities[IDX(u, v, g->n)] - flowMatrix[IDX(u, v, g->n)];
      if ((residual > 0) && (parents[v] == -1))
      {
        parents[v] = u;
        pathCapacities[v] = std::min(pathCapacities[u], residual);
        if (v != t)
        {
          bfsQueue.push(v);
        }
        else
        {
          int result = pathCapacities[t];
          return result;
        }
      }
    }
  }
  return 0;
}

// Edmonds-Karp algorithm to find max s-t flow
Flow *edKarpSeq(Graph *g, int s, int t)
{
  int flow = 0;
  int *flowMatrix = (int *)calloc((g->n * g->n), sizeof(int));
  int *parents = (int *)malloc(g->n * sizeof(int));
  int *pathCapacities = (int *)calloc(g->n, sizeof(int));
  while (true)
  {
    int tempCapacity = sequentialBFS(g, flowMatrix, parents, pathCapacities, s, t);
    if (tempCapacity == 0)
    {
      break;
    }
    flow += tempCapacity;
    int v = t;
    // backtrack
    while (v != s)
    {
      int u = parents[v];
      flowMatrix[IDX(u, v, g->n)] += tempCapacity;
      flowMatrix[IDX(v, u, g->n)] -= tempCapacity;
      v = u;
    }
  }
  Flow *result = (Flow *)malloc(sizeof(Flow));
  result->maxFlow = flow;
  result->finalEdgeFlows = flowMatrix;
  free(parents);
  free(pathCapacities);
  return result;
}

// Subroutine for Dinic's algorithm, used to update the levels of nodes in the
// layered graph with BFS. Returns true if a path from s to t with positive
// capacity still exists.
bool bfsLayeredGraph(Graph *g, int *flowMatrix, int *levels, int s, int t)
{
  memset(levels, -1, (g->n * sizeof(int)));
  levels[s] = 0;
  std::queue<int> bfsQueue;
  bfsQueue.push(s);
  while (!bfsQueue.empty())
  {
    int u = bfsQueue.front();
    bfsQueue.pop();
    for (int v = 0; v < g->n; v++)
    {
      if (u == v)
        continue;
      int residual = g->capacities[IDX(u, v, g->n)] - flowMatrix[IDX(u, v, g->n)];
      if ((residual > 0) && (levels[v] == -1))
      {
        levels[v] = levels[u] + 1;
        if (v != t)
        {
          bfsQueue.push(v);
        }
        else
        {
          return true;
        }
      }
    }
  }
  return false;
}

// DFS routine to find blocking flow
int dinicSearch(Graph *g, int *flowMatrix, int *levels, int *curNeighbor,
                int u, int t, int curFlow)
{
  if (u == t)
    return curFlow;
  for (; curNeighbor[u] < g->n; curNeighbor[u]++)
  {
    int v = curNeighbor[u];
    int residual = g->capacities[IDX(u, v, g->n)] - flowMatrix[IDX(u, v, g->n)];
    if ((residual > 0) && (levels[v] == (levels[u] + 1)))
    {
      int newFlow = std::min(residual, curFlow);
      int actualFlow = dinicSearch(g, flowMatrix, levels, curNeighbor, v, t, newFlow);
      if (actualFlow > 0)
      {
        flowMatrix[IDX(u, v, g->n)] += actualFlow;
        flowMatrix[IDX(v, u, g->n)] -= actualFlow;
        return actualFlow;
      }
    }
  }
  return 0;
}

// Dinic's algorithm to find max s-t flow
Flow *dinicSeq(Graph *g, int s, int t)
{
  int flow = 0;
  int maxInt = std::numeric_limits<int>::max();
  int *curNeighbor = (int *)malloc(g->n * sizeof(int));
  int *levels = (int *)malloc(g->n * sizeof(int));
  int *flowMatrix = (int *)calloc((g->n * g->n), sizeof(int));
  while (true)
  {
    memset(curNeighbor, 0, (g->n * sizeof(int)));
    bool existsPath = bfsLayeredGraph(g, flowMatrix, levels, s, t);
    if (!existsPath)
    {
      break;
    }
    while (true)
    {
      int tempFlow = dinicSearch(g, flowMatrix, levels, curNeighbor, s, t, maxInt);
      if (tempFlow == 0)
      {
        break;
      }
      flow += tempFlow;
    }
  }
  Flow *result = (Flow *)malloc(sizeof(Flow));
  result->maxFlow = flow;
  result->finalEdgeFlows = flowMatrix;
  free(curNeighbor);
  free(levels);
  return result;
}
