#include <tuple>
#include <vector>
#include "graphs.h"

struct FlowGraphResult
{
  int flow;
  std::vector<std::tuple<int, int, int>> *flowEdges;
};

class FlowGraph
{
  int n;

public:
  FlowGraph(int);
  ~FlowGraph();
  void SetS(int);
  void SetT(int);
  void AddEdge(int, int, int);
  void DeleteEdge(int, int);
  void AddEdges(std::vector<std::tuple<int, int, int>> *);
  void DeleteEdges(std::vector<std::tuple<int, int>> *);
  FlowGraphResult *FlowEdmondsKarp();
  FlowGraphResult *FlowEdmondsKarp(std::string);
  FlowGraphResult *FlowDinic();
  FlowGraphResult *FlowDinic(std::string);

private:
  FlowGraphResult *ProcessResult(Flow *);
  int *capacities;
  int s;
  int t;
};
