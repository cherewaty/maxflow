# maxflow

Parallel Edmonds-Karp and Dinic's max flow algorithms in C++, with CUDA.

## Build

### `make`

Build an executable (`maxflow`) to run both the sequential and GPU-enabled parallel versions of the algorithms. Requires CUDA.

### `make -f sequential-only.mk`

Build an executable (`maxflow-sequential`) to run only the sequential versions of the algorithms.
