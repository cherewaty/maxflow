# maxflow

Parallel Edmonds-Karp and Dinic's max flow algorithms in C++, with CUDA.

## Build

### `make -f sequential.mk`

Build an executable to run only the sequential versions of the algorithms.

### `make -f gpu.mk`

Build an executable to run both the sequential and GPU-enabled parallel versions of the algorithms. Requires CUDA.
