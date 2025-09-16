CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Harry Guan
* (TODO) [LinkedIn]()
* Tested on:  Windows 11, Intel i7-14700 @ 2.10GHz 32GB, NVIDIA T1000 4GB (Moore 100B virtual labs)

### Project Overview
This project implements GPU stream compaction algorithms in CUDA, focusing on the fundamental parallel algorithms of scan (prefix sum) and stream compaction. 
Stream compaction is a crucial operation that removes unwanted elements (zeros in this case) from an array while maintaining the relative order of the remaining elements. 
This algorithm is essential for GPU path tracing and other parallel algorithms.

Example:
```
Input:  [1, 5, 0, 1, 2, 0, 3]
Flags:  [1, 1, 0, 1, 1, 0, 1]
Scan:   [0, 1, 2, 2, 3, 4, 4]
Output: [1, 5, 1, 2, 3]
```

### Features Implemented
- **CPU Implementations**:
  - **scan**: Exclusive prefix sum using simple for loop
  - **compactWithoutScan**: Stream compaction without using scan
  - **compactWithScan**: Stream compaction using scan and scatter


### Naive Scan

A algorithm that uses global memory exclusively. In each of `log(n)` passes, it adds elements together at increasing strides.

- **Approach**: Launches a thread per element, adds pairs with increasing offsets.
- **Time Complexity**: `O(n log n)`
- **Limitation**: A separate kernel launch per pass introduces **serialization overhead**, reducing performance.

###  Work-Efficient Scan

A more optimized approach that reduces redundant computations and kernel launches. It processes the data in two phases:

#### Up-Sweep Phase
- Builds a balanced binary tree of partial sums.

#### Down-Sweep Phase
- Traverses the tree in reverse to compute the exclusive scan results.

- **Time Complexity**: `O(n)`
- **Advantage**: Significantly more efficient than the naive version.

### 4. Thrust Scan

A wrapper around the **Thrust** library’s `exclusive_scan`:

- Provides a benchmark for performance comparison.
- Highly optimized and production-ready.

Stream compaction follows a three-step process:

1. **Map**: Convert input array to boolean flags (1 for non-zero, 0 for zero)
2. **Scan**: Compute exclusive prefix sum of the boolean array
3. **Scatter**: Place non-zero elements at positions given by the scan results

## Performance Analysis
	
## Block size optimization
| Block size | naive scan, power-of-two (ms) | naive scan, non-power-of-two (ms) | work-efficient scan, power-of-two (ms) | work-efficient scan, non-power-of-two (ms) |
| :--- | :--- | :--- | :--- | :--- |
| 128 | 5.36 | 5.4 | 2.37 | 2.34 |
| 256 | 6.05 | 5.4 | 2.6 | 2.39 |
| 512 | 5.7 | 6.17 | 2.61 | 2.5 |
| 1024 | 5.85 | 6.39 | 2.71 | 2.56 |


Having a look at the run time, tt seem like block size does not have a significant impact on performance. But the best performance for naive scan is at block size 128, while work-efficient scan performs best at block size 128 as well. So we would examine our tests using block size of 128 for both Work efficient and Naive.

## Size of an array runtime analysis 
![](img/graph1.png)
![](img/graph2.png)


Analysis: 

### Test Results
```
****************
** SCAN TESTS **
****************
    [  35   9  39   5  34   1  41  13  45  43  23  44  20 ...  45   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 1.7865ms    (std::chrono Measured)
    [   0  35  44  83  88 122 123 164 177 222 265 288 332 ... 25689533 25689578 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 1.8431ms    (std::chrono Measured)
    [   0  35  44  83  88 122 123 164 177 222 265 288 332 ... 25689405 25689454 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 6.30218ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 5.49034ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 8.95414ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 8.35494ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 39.2416ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 32.6714ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   1   3   3   0   1   3   1   3   1   1   0   2 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 3.4072ms    (std::chrono Measured)
    [   1   1   3   3   1   3   1   3   1   1   2   3   2 ...   2   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 3.43ms    (std::chrono Measured)
    [   1   1   3   3   1   3   1   3   1   1   2   3   2 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 9.6884ms    (std::chrono Measured)
    [   1   1   3   3   1   3   1   3   1   1   2   3   2 ...   2   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 6.65181ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 6.12739ms    (CUDA Measured)
    passed
```