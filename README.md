The resultant of two vectors, calculated by adding respective components of
each vector is called vector sum.

```c
Each thread computes the sum of a single component of vector. Since there
are only 10 components, but 12 total threads, we must ensure to not access
out of bounds (last 2 threads wont compute anything).
threadIdx.x: thread index, within block (0 ... 3)
blockIdx.x:  block index, within grid (0 ... 2)
blockDim.x:  number of threads in a block (4)
i: index into the vectors
```

```c
1. Allocate space for 3 vectors A, B, and C (of length 10).
2. Define vectors A and B (C = A + B will be computed by GPU).
3. Allocate space for A, B, C on GPU.
4. Copy A, B from host memory to device memory (GPU).
5. Execute kernel with 4 threads per block, and 3 block (4*3 >= 10).
6. Wait for kernel to complete, and copy C from device to host memory.
7. Validate if the vector sum is correct (on CPU).
```

```bash
# OUTPUT
a = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18}
b = {0, -1, -2, -3, -4, -5, -6, -7, -8, -9}
c = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

See [main.cu] for code.

[main.cu]: main.cu


### references

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
