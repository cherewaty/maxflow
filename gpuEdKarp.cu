#include "gpuEdKarp.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IDX(i, j, n) ((i) * (n) + (j))

__global__ void backTrack(int *parents,int *flowMatrix, int s,int v,int tempCapacity,int n){
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if(index < n){
    if (index != s ){
      int u = parents[index];
      atomicAdd(&flowMatrix[IDX(u,index,n)],tempCapacity);
      atomicSub(&flowMatrix[IDX(index,u,n)],tempCapacity);
    }    
  }

}

/*
*   Source from https://github.com/vulq/Flo
*   Update by Roberto and Jefferey
*/

int BFS(Graph *g, int *flowMatrix, int *parents, int *pathCapacities, int s, int t)
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
Flow *edKarpGpu(Graph *g, int s, int t){
  int sizeN = g->n;
  int flow = 0;
  int *flowMatrix = (int *)calloc((sizeN * sizeN), sizeof(int));
  int *parents = (int *)malloc(sizeN * sizeof(int));
  int *pathCapacities = (int *)calloc(sizeN, sizeof(int));

  int *d_flowMaxtrix;
  int *d_parents;
  

  //Code example https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels
  //int blockSize; // The launch configurator returned block size 
  //int minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
  //int gridSize; // The actual grid size needed, based on input size 
  //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, backTrack, 0, sizeN);
  //gridSize = (sizeN + blockSize - 1) / blockSize; 

  cudaMalloc((void **)&d_flowMaxtrix,sizeN * sizeN * sizeof(int));
  cudaMalloc((void **)&d_parents,sizeN * sizeof(int));

  while (true)
  {
    int tempCapacity = BFS(g, flowMatrix, parents, pathCapacities, s, t);
    if (tempCapacity == 0)
    {
      break;
    }
    flow += tempCapacity;
    int v = t;
    // backtrack
    //copy from host(my computer) to device(GPU)
    cudaMemcpy(d_flowMaxtrix,flowMatrix,sizeN * sizeN* sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_parents,parents,sizeN* sizeof(int),cudaMemcpyHostToDevice);
    // backtrack

    //backTrack<<<gridSize,blockSize>>>(d_parents,d_flowMaxtrix,s,v,tempCapacity,sizeN);
    backTrack<<<sizeN/32,32>>>(d_parents,d_flowMaxtrix,s,v,tempCapacity,sizeN);
    //copy device to host
    cudaMemcpy(flowMatrix,d_flowMaxtrix,sizeN * sizeN * sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(parents,d_parents,sizeN * sizeof(int),cudaMemcpyDeviceToHost);
  }
  Flow *result = (Flow *)malloc(sizeof(Flow));
  result->maxFlow = flow;
  result->finalEdgeFlows = flowMatrix;
  free(parents);
  free(pathCapacities);
  cudaFree(d_flowMaxtrix);
  cudaFree(d_parents);
  return result;
}
