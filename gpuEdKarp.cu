#include "gpuEdKarp.h"
#include <algorithm>
#include <limits>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define UPDIV(n, d)   (((n) + (d) - 1) / (d))
#define IDX(i, j, n) ((i) * (n) + (j))
static dim3 threadsPerBlock(1024, 1, 1);

__global__ void backTrack(int *parents,int *flowMatrix, int s,int v,int tempCapacity,int n){
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if(index < n){
    if (index != s ){
      int u = parents[index];
      if(u > -1){
        atomicAdd(&flowMatrix[IDX(u,index,n)],tempCapacity);
        atomicSub(&flowMatrix[IDX(index,u,n)],tempCapacity);
      }
    }    
  }
  __syncthreads();
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
  while (!bfsQueue.empty()){
    int u = bfsQueue.front();
    bfsQueue.pop();
    for (int v = 0; v < g->n; v++){
      if (u == v){
        continue;
      }
      printf("Loop U %d , V %d\n",u,v);
      printf("flowmatrix bfs %d\n",flowMatrix[IDX(u, v, g->n)]);
      printf("cap bfs %d\n",g->capacities[IDX(u, v, g->n)]);
      int residual = g->capacities[IDX(u, v, g->n)] - flowMatrix[IDX(u, v, g->n)];
      if ((residual > 0) && (parents[v] == -1)){
        printf("parents u %d v %d \n%",u,v);
        parents[v] = u;
        pathCapacities[v] = std::min(pathCapacities[u], residual);
        if (v != t){
          bfsQueue.push(v);
        }else{
          int result = pathCapacities[t];
          return result;
        }
      }
    }
  }
  printf("pathCap %d\n",pathCapacities[t]);
  return 0;
}

// Edmonds-Karp algorithm to find max s-t flow
Flow *edKarpGpu(Graph *g, int s, int t){
  int sizeN = g->n;
  int flow = 0;
  int *flowMatrix = (int *)malloc((sizeN * sizeN) * sizeof(int));
  int *parents = (int *)malloc(sizeN * sizeof(int));
  int *pathCapacities = (int *)malloc(sizeN * sizeof(int));

  int *d_flowMaxtrix;
  int *d_parents;

  cudaMalloc((void **)&d_flowMaxtrix,sizeN * sizeN * sizeof(int));
  cudaMalloc((void **)&d_parents,sizeN * sizeof(int));

  while (true){
    int tempCapacity = BFS(g, flowMatrix, parents, pathCapacities, s, t);
    printf("temp cap %d\n",tempCapacity);
    if (tempCapacity == 0){
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
    int numBlocks = UPDIV((sizeN-2),threadsPerBlock.x);
    backTrack<<<numBlocks,threadsPerBlock>>>(d_parents,d_flowMaxtrix,s,v,tempCapacity,sizeN);
    
    cudaThreadSynchronize();
    //copy device to host
    cudaMemcpy(flowMatrix,d_flowMaxtrix,sizeN * sizeN * sizeof(int),cudaMemcpyDeviceToHost);
    //cudaMemcpy(parents,d_parents,sizeN * sizeof(int),cudaMemcpyDeviceToHost);
  }
  Flow *result = (Flow *)malloc(sizeof(Flow));
  result->maxFlow = flow;
  result->finalEdgeFlows = flowMatrix;
  free(parents);
  free(pathCapacities);
  free(flowMatrix);
  cudaFree(d_flowMaxtrix);
  cudaFree(d_parents);
  return result;
}
