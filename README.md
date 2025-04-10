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
$ nvcc -std=c++17 -Xcompiler -O3 main.cu
$ ./a.out

# CPU vector-sum ...
# Execution time: 0.0 ms
#
# GPU vector-sum, simple ...
# Execution time: 6.2 ms
#
# GPU vector-sum, multiple ...
# Execution time: 6.1 ms
#
# GPU vector-sum, chunked with stream ...
# Execution time: 7.5 ms
#
# GPU vector-sum, chunked with streams ...
# Execution time: 7.6 ms
#
# GPU vector-sum, chunked with interleaved streams ...
# Execution time: 6.7 ms
```

See [main.cu] for code.

[main.cu]: main.cu

<br>
<br>


## References

- [CUDA by Example :: Jason Sanders, Edward Kandrot](https://gist.github.com/wolfram77/72c51e494eaaea1c21a9c4021ad0f320)

![](https://ga-beacon.deno.dev/G-G1E8HNDZYY:v51jklKGTLmC3LAZ4rJbIQ/github.com/moocf/vector-sum.cuda)
