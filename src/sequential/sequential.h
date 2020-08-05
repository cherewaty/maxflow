#include "../graphs.h"

/*
*   Source from https://github.com/vulq/Flo
*/

Flow *edKarpSeq(Graph *g, int s, int t);

Flow *dinicSeq(Graph *g, int s, int t);

int dinicSearch(Graph *g, int *flowMatrix, int *levels, int *curNeighbor,
                int u, int t, int curFlow);
