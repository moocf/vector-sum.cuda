The resultant of two vectors, calculated by adding respective components of
each vector is called vector sum.

```c
main():
1. Allocate space for 3 vectors A, X, Y (of length 2000000).
2. Define vectors X and Y (A = X + Y will be computed).
3. Calculate expected value for varifying results.
4. Run vector sum on with various approaches.
5. Free vectors A, X, Y.
```

```bash
# OUTPUT
CPU vector-sum ...
Execution time: 6.0 ms

GPU vector-sum, simple ...
Execution time: 6.7 ms

GPU vector-sum, multiple ...
Execution time: 7.2 ms

GPU vector-sum, chunked with stream ...
Execution time: 8.9 ms

GPU vector-sum, chunked with streams ...
Execution time: 9.5 ms

GPU vector-sum, chunked with interleaved streams ...
Execution time: 7.2 ms
```

See [main.cu] for code, [main.ipynb] for notebook.

[main.cu]: main.cu
[main.ipynb]: https://colab.research.google.com/drive/1d8TouY8FdadWOxPPWVjemS5hekFUTqzB?usp=sharing


### references

- [CUDA by Example :: Jason Sanders, Edward Kandrot](http://www.mat.unimi.it/users/sansotte/cuda/CUDA_by_Example.pdf)
