#include <stdio.h>
#include "main.h"
#include "_host.h"
#include "_simple.h"


// 1. Allocate space for 3 vectors A, X, Y (of length 1000000).
// 2. Define vectors A and B (A = X + Y will be computed by GPU).
// 3. Allocate space for A, B, C on GPU.
// 4. Copy A, B from host memory to device memory (GPU).
// 5. Execute kernel with 4 threads per block, and 3 block (4*3 >= 10).
// 6. Wait for kernel to complete, and copy C from device to host memory.
// 7. Validate if the vector sum is correct (on CPU).
int main() {
  int N = 1000000;                 // 1
  size_t N1 = N * sizeof(int);     // 1

  float *a = (float*) malloc(N1);  // 1
  float *x = (float*) malloc(N1);  // 1
  float *y = (float*) malloc(N1);  // 1
  for (int i=0; i<N; i++) { // 2
    x[i] = (float) 2*i;     // 2
    y[i] = (float) -i;      // 2
  }                         // 2

  float duration;
  printf("CPU vector-sum ...\n");
  duration = run_host(x, y, y, N);
  printf("Execution time: %3.1f ms\n", duration);
  printf("\n");

  float combined = vector_combined(a, N);
  printf("GPU vector-sum, simple ...\n");
  duration = 

  
  float *aD, *bD, *cD;        // 3
  TRY( cudaMalloc(&aD, N1) ); // 3
  TRY( cudaMalloc(&bD, N1) ); // 3
  TRY( cudaMalloc(&cD, N1) ); // 3
  TRY( cudaMemcpy(aD, a, N1, cudaMemcpyHostToDevice) ); // 4
  TRY( cudaMemcpy(bD, b, N1, cudaMemcpyHostToDevice) ); // 4

  int threads = 4;                            // 5
  int blocks  = CEILDIV(N, threads);          // 5
  kernel<<<blocks, threads>>>(cD, aD, bD, N); // 5

  TRY( cudaMemcpy(c, cD, N1, cudaMemcpyDeviceToHost) ); // 6
  printf("a = "); PRINTVEC(a, N); printf("\n");
  printf("b = "); PRINTVEC(b, N); printf("\n");
  printf("c = "); PRINTVEC(c, N); printf("\n");

  for (int i=0; i<N; i++) {  // 7
    if (c[i] == i) continue; // 7
    fprintf(stderr, "%f + %f != %f (at component %d)\n", a[i], b[i], c[i], i);
  }
  return 0;
}
