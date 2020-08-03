#include <tuple>
#include <vector>
#include "flow_graph.h"
#include "graphs.h"
#include "sequential.h"

#define IDX(i, j, n) ((i) * (n) + (j))

/*
*   Signifcant portions of source from https://github.com/vulq/Flo
*/

FlowGraphResult *FlowGraph::ProcessResult(Flow *result)
{
  FlowGraphResult *finalResult = (FlowGraphResult *)malloc(sizeof(FlowGraphResult));
  finalResult->flow = result->maxFlow;
  std::vector<std::tuple<int, int, int>> *flowEdges = new std::vector<std::tuple<int, int, int>>;
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < n; j++)
    {
      int cap = result->finalEdgeFlows[IDX(i, j, n)];
      if (cap > 0)
      {
        (*flowEdges).push_back(std::make_tuple(i, j, cap));
      }
    }
  }
  finalResult->flowEdges = flowEdges;
  free(result->finalEdgeFlows);
  free(result);
  return finalResult;
}

FlowGraphResult *FlowGraph::FlowEdmondsKarp()
{
  Graph *g = (Graph *)malloc(sizeof(Graph));
  g->n = n;
  g->capacities = capacities;
  Flow *result;

  result = edKarpSeq(g, s, t);

  free(g);
  return ProcessResult(result);
}

FlowGraphResult *FlowGraph::FlowDinic()
{
  Graph *g = (Graph *)malloc(sizeof(Graph));
  g->n = n;
  g->capacities = capacities;
  Flow *result;

  result = dinicSeq(g, s, t);

  free(g);
  return ProcessResult(result);
}

FlowGraph::FlowGraph(int a)
{
  n = a;
  capacities = (int *)calloc((n * n), sizeof(int));
  s = 0;
  t = n - 1;
}

FlowGraph::~FlowGraph()
{
  free(capacities);
}

void FlowGraph::SetS(int a)
{
  s = a;
}

void FlowGraph::SetT(int a)
{
  t = a;
}

void FlowGraph::AddEdge(int u, int v, int cap)
{
  if ((0 <= u) && (u < n) && (0 <= v) && (v < n))
  {
    capacities[IDX(u, v, n)] = cap;
  }
}

void FlowGraph::DeleteEdge(int u, int v)
{
  if ((0 <= u) && (u < n) && (0 <= v) && (v < n))
  {
    capacities[IDX(u, v, n)] = 0;
  }
}

void FlowGraph::AddEdges(std::vector<std::tuple<int, int, int>> *edges)
{
  for (std::vector<std::tuple<int, int, int>>::iterator it = (*edges).begin(); it != (*edges).end(); it++)
  {
    int u = std::get<0>(*it);
    int v = std::get<1>(*it);
    int cap = std::get<2>(*it);
    AddEdge(u, v, cap);
  }
}

void FlowGraph::DeleteEdges(std::vector<std::tuple<int, int>> *edges)
{
  for (std::vector<std::tuple<int, int>>::iterator it = (*edges).begin(); it != (*edges).end(); it++)
  {
    int u = std::get<0>(*it);
    int v = std::get<1>(*it);
    DeleteEdge(u, v);
  }
}
