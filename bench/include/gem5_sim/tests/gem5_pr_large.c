/*
 * gem5 Cache Policy Test: PageRank-like workload that overflows L3.
 *
 * Working set:
 *   scores[]:  NUM_VERTICES × 4B = 1MB  (256K vertices)
 *   contrib[]: NUM_VERTICES × 4B = 1MB
 *   edges[]:   NUM_EDGES × 4B   = 8MB   (2M edges)
 *   offsets[]: NUM_VERTICES × 4B = 1MB
 *   Total:     ~11MB → overflows 8MB L3 cache
 *
 * Access pattern:
 *   - Sequential scan of offsets[] and edges[] (streaming, benefits from stride)
 *   - Irregular scatter/gather on scores[] and contrib[] via edges[i]
 *   - Power-law degree distribution: 10% of edges go to top 1% vertices (hubs)
 *   - This creates the reuse imbalance that GRASP/ECG exploit
 *
 * Expected policy behavior:
 *   - LRU: treats all lines equally, hubs get evicted despite high reuse
 *   - SRRIP: scan-resistant but no degree awareness
 *   - GRASP: protects hub vertices (low RRPV), evicts cold faster
 *   - ECG: layered tiebreaking, should match or beat GRASP
 *
 * Compile: gcc -O1 -static -o gem5_pr_large gem5_pr_large.c -lm
 * Run:     gem5.opt config.py (see gem5_policy_sweep.py)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_VERTICES (256 * 1024)       /* 256K vertices */
#define NUM_EDGES    (2 * 1024 * 1024)  /* 2M edges */
#define ITERATIONS   3

int main() {
    float *scores  = (float*)calloc(NUM_VERTICES, sizeof(float));
    float *contrib = (float*)calloc(NUM_VERTICES, sizeof(float));
    int   *edges   = (int*)malloc(NUM_EDGES * sizeof(int));
    int   *offsets = (int*)malloc((NUM_VERTICES + 1) * sizeof(int));

    if (!scores || !contrib || !edges || !offsets) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Initialize scores */
    for (int i = 0; i < NUM_VERTICES; i++)
        scores[i] = 1.0f / NUM_VERTICES;

    /* Generate edges with power-law-like skew (hub vertices) */
    srand(42);
    for (int i = 0; i < NUM_EDGES; i++) {
        if (rand() % 10 == 0)
            edges[i] = rand() % (NUM_VERTICES / 100);  /* 10% edges → top 1% */
        else
            edges[i] = rand() % NUM_VERTICES;
    }

    /* Uniform degree distribution for offsets (simplification) */
    for (int i = 0; i <= NUM_VERTICES; i++)
        offsets[i] = (int)((long)i * NUM_EDGES / NUM_VERTICES);

    /* PageRank-like iterations with irregular access */
    for (int iter = 0; iter < ITERATIONS; iter++) {
        memset(contrib, 0, NUM_VERTICES * sizeof(float));

        for (int v = 0; v < NUM_VERTICES; v++) {
            int start = offsets[v];
            int end   = offsets[v + 1];
            if (end <= start) continue;

            float c = scores[v] / (end - start);
            for (int e = start; e < end; e++)
                contrib[edges[e]] += c;  /* Irregular scatter */
        }

        for (int v = 0; v < NUM_VERTICES; v++)
            scores[v] = 0.15f / NUM_VERTICES + 0.85f * contrib[v];
    }

    /* Print result to prevent dead-code elimination */
    float total = 0;
    for (int i = 0; i < NUM_VERTICES; i++)
        total += scores[i];

    printf("PR sum: %f (V=%d E=%d iters=%d)\n",
           total, NUM_VERTICES, NUM_EDGES, ITERATIONS);

    free(scores);
    free(contrib);
    free(edges);
    free(offsets);
    return 0;
}
