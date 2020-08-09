#include "gpuEdKarp.h"




__device__ int indexFinder(int i,int n, int j){
    return ((i) * (n) + (j));
}

__global__ void backTrack(int *parents,int *flowMatrix, int s,int v,int tempCapacity,int n){
    while (v != s){
        int u = parents[v];
        flowMatrix[indexFinder(u,v,n)] += tempCapacity;
        flowMatrix[indexFinder(v,u,n)] -= tempCapacity;
        v = u;
      }
}

__global__ void nextQueue(int queueSize,int *addedToQueue,bool *hasResult,int *result,int *queue,int *queueIndex int *parents, int *pathCapacities,int u,int n,int *capacities, int *flowMatrix){
    *addedToQueue = 0;
    for (int v = 0; v < n; v++){
        if (u == v){
          continue;
        }
        int residual = capacities[indexFinder(u, v, n)] - flowMatrix[indexFinder(u, v, n)];
        if ((residual > 0) && (parents[v] == -1)){
          parents[v] = u;
          pathCapacities[v] = fminf(pathCapacities[u], residual);
          if (v != t){
            queue[queueSize + *addedToQueue]=v;
            *addedToQueue++;
          }else{
            *hasResult = true;
            *result = pathCapacities[t];
            break;
          }
        }
      }
}

//Code Reference https://github.com/kaletap/bfs-cuda-gpu/blob/master/src/gpu/simple/bfs_simple.cu
int BFS(Graph *g, int *flowMatrix, int *parents, int *pathCapacities, int s, int t,int *d_parents,int *d_flowMaxtrix,int *d_pathCapacities){
  memset(parents, -1, (g->n * sizeof(int)));
  memset(pathCapacities, 0, (g->n * sizeof(int)));
  parents[s] = s;
  pathCapacities[s] = std::numeric_limits<int>::max();

  int currentQueueSize = 1;

  int *queue;
  int queueIndex = 0;
  int addedToQueue = 0;
  int result = 0;
  
  int *d_queue;
  int *d_queueIndex;
  int *d_addedToQueue;
  int *d_result;
  bool *d_hasResult;
  int size = g->n * sizeof(int);

  malloc(queue,size);
  cudaMalloc((void **)&d_queue, size);
  cudaMalloc((void**)&d_queueIndex,sizeof(int));
  cudaMalloc((void**)&d_addedToQueue,sizeof(int));
  cudaMalloc((void**)&d_hasResult,sizeof(bool));
  cudaMalloc((void**)&d_result,sizeof(int));
  *hasResult = false;
  queue[0] = s;

  while(currentQueueSize > 0){
    int u = queue[queueIndex];
    cudaMemcpy(d_hasResult,hasResult,sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(d_resutl,result,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_addedToQueue,addedToQueue,sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_queue,queue,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_queueIndex,queueIndex,sizeof(int),cudaMemcpyHostToDevice);

    nextQueue<<<1,1>>>(currentQueueSize,d_addedToQueue,d_hasResult,d_result,d_queue,d_queueIndex,d_parents,d_pathCapacities,u,g->n,d_capacities,d_flowMaxtrix);
    cudaMemcpy(hasResult,d_hasResult,sizeof(bool),cudaMemcpyDeviceToHost);
    cudaMemcpy(result,d_resutl,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(addedToQueue,d_addedToQueue,sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(parents,d_parents,size,cudaMemcpyDeviceToHost);
    cudaMemcpy(pathCapacities,d_pathCapacities,size,cudaMemcpyDeviceToHost);
    if(hasResult){
        free(queue)
        cudaFree(d_hasResult);
        cudaFree(d_result);
        cudaFree(d_queueIndex);
        cudaFree(d_addedToQueue);
        cudaFree(d_queue);
        return result
    }else {
        cudaMemcpy(queue,d_queue,size,cudaMemcpyDeviceToHost);
        currentQueueSize = addedToQueue -1;
        queueIndex++;
    }

  }
  free(queue)
  cudaFree(d_hasResult);
  cudaFree(d_result);
  cudaFree(d_queueIndex);
  cudaFree(d_addedToQueue);
  cudaFree(d_queue);
  return 0;
}





Flow *edKarpGpu(Graph *g, int s, int t){
    int flow = 0;
    int *flowMatrix = (int *)malloc(g->n * g->n * sizeof(int));
    int *parents = (int *)malloc(g->n * sizeof(int));
    int *pathCapacities = (int *)malloc(g->n* sizeof(int));
    
    
    int *d_flowMaxtrix;
    int *d_parents;
    int *d_pathCapacities;

    cudaMalloc((void **)&d_flowMaxtrix,g->n * g->n * sizeof(int));
    cudaMalloc((void **)&d_parents,g->n * sizeof(int));
    cudaMalloc((void **)&d_pathCapacities,g->n* sizeof(int));

    while (true){
      int tempCapacity = BFS(g, flowMatrix, parents, pathCapacities, s, t,d_pathCapacities,d_flowMaxtrix,d_pathCapacities);
      if (tempCapacity == 0){
        break;
      }
      flow += tempCapacity;
      int v = t;

      //copy from host(my computer) to device(GPU)
      cudaMemcpy(d_flowMaxtrix,flowMatrix,g->n * g->n * sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(d_parents,parents,g->n * sizeof(int),cudaMemcpyHostToDevice);
      // backtrack

      backTrack<<<1,1,>>>(d_parents,d_flowMaxtrix,s,v,tempCapacity,g_>n);
      //copy device to host
      cudaMemcpy(flowMatrix,d_flowMaxtrix,g->n * g->n * sizeof(int),cudaMemcpyDeviceToHost);
      cudaMemcpy(parents,d_parents,g->n * sizeof(int),cudaMemcpyDeviceToHost);
    }
    Flow *result = (Flow *)malloc(sizeof(Flow));
    result->maxFlow = flow;
    result->finalEdgeFlows = flowMatrix;
    free(parents);
    free(pathCapacities);

    cudaFree(d_flowMaxtrix);
    cudaFree(d_parents);
    cudaFree(d_pathCapacities);
    return result;


Flow *dinicGpu(Graph *g, int s, int t){
  // TODO: implement this
}