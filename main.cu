#include <stdio.h>
#include "main.h"
#include "_host.h"
#include "_simple.h"
#include "_multiple.h"
#include "_stream.h"
#include "_streams.h"
#include "_streams_interleaved.h"


// 1. Allocate space for 3 vectors A, X, Y (of length 2000000).
// 2. Define vectors X and Y (A = X + Y will be computed).
// 3. Calculate expected value for varifying results.
// 4. Run vector sum on with various approaches.
// 5. Free vectors A, X, Y.
int main() {
  int N = 2000000;
  int CHUNK = 65536;
  size_t N1 = N * sizeof(float);

  float *a = (float*) malloc(N1);  // 1
  float *x = (float*) malloc(N1);  // 1
  float *y = (float*) malloc(N1);  // 1
  for (int i=0; i<N; i++) { // 2
    x[i] = i * 0.2f;        // 2
    y[i] = i * -0.1f;       // 2
  }

  vector_sum(a, x, y, N);      // 3
  float exp = SUM_ARRAY(a, N); // 3

  printf("CPU vector-sum ...\n");            // 4
  printrun(exp, a, N, run_host(a, x, y, N)); // 4
  
  printf("GPU vector-sum, simple ...\n");      // 4
  printrun(exp, a, N, run_simple(a, x, y, N)); // 4

  printf("GPU vector-sum, multiple ...\n");      // 4
  printrun(exp, a, N, run_multiple(a, x, y, N)); // 4

  printf("GPU vector-sum, chunked with stream ...\n"); // 4
  printrun(exp, a, N, run_stream(a, x, y, N, CHUNK));  // 4

  printf("GPU vector-sum, chunked with streams ...\n"); // 4
  printrun(exp, a, N, run_streams(a, x, y, N, CHUNK));  // 4

  printf("GPU vector-sum, chunked with interleaved streams ...\n"); // 4
  printrun(exp, a, N, run_streams_interleaved(a, x, y, N, CHUNK));  // 4

  free(y); // 5
  free(x); // 5
  free(a); // 5
  return 0;
}
