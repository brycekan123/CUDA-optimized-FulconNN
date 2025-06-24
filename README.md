# CUDA Neural Network Forward and Backward Pass Optimization

A comprehensive implementation demonstrating GPU acceleration techniques for neural network training, achieving **~2x speedup** through CUDA optimization strategies.
## 🚀 Project Overview

This project implements and optimizes both forward and backward passes of a neural network layer using CUDA. Starting from naive implementations, I progressively applied GPU optimization techniques including **vectorization**, **tiling**, and **shared memory** management to reduce execution time.

### Key Features
- **Forward Pass**: Matrix multiplication with bias addition and sigmoid activation
- **Backward Pass**: Gradient computation for weights and biases
- **Progressive Optimization**: From naive to highly optimized implementations
- **Performance Benchmarking**: Comprehensive timing and verification

## 🎯 Learning Objectives & Achievements

### Core GPU Optimization Techniques 

#### 1. **Vectorization** 
- Implemented `float4` vectorized memory operations for coalesced access
- Applied to both forward pass input loading and backward pass bias computation

```cuda
// Vectorized memory access example
float4 vecX = *reinterpret_cast<const float4*>(&X[globalRow * input_dim + globalCol]);
X_tile[threadRow * TM + m][tileCol + 0] = vecX.x;
X_tile[threadRow * TM + m][tileCol + 1] = vecX.y;
X_tile[threadRow * TM + m][tileCol + 2] = vecX.z;
X_tile[threadRow * TM + m][tileCol + 3] = vecX.w;
```

#### 2. **Tiling Strategy**
- Implemented blocked matrix multiplication to maximize data reuse
- Used shared memory tiles to reduce global memory accesses
- Optimized tile dimensions: `BLOCK_SIZE * 4` for enhanced parallelism

#### 3. **Shared Memory Optimization**
- Strategic use of `__shared__` memory to cache frequently accessed data
- Implemented coalesced loading patterns for optimal memory throughput

#### 4. **Thread-Level Parallelism**
- Each thread computes multiple results (`TM=4, TN=2` → 8 results per thread)
- Maximized computational intensity per thread

## 🏗️ Architecture & Implementation

### Forward Pass Optimization Journey

1. **Naive Implementation**
   - Basic thread-per-element approach
   - Direct global memory access
   - Simple but inefficient

2. **Optimized Implementation**
   - **Tiling**: Block-based computation with shared memory
   - **Vectorization**: `float4` operations for memory coalescing  
   - **Thread Workload**: Each thread handles 4×2 output elements
   - **Memory Hierarchy**: Strategic use of shared memory caching

### Backward Pass Optimization Challenges

The backward pass presented unique optimization challenges:

- **Weight Gradients**: Required X^T × dL/dZ computation
- **Memory Access Patterns**: Column-wise access prevented full vectorization
- **Solution**: Focused on tiling and increased per-thread workload
- **Bias Gradients**: Successfully implemented full vectorization

### Key Technical Decisions

```cuda
// Optimized thread configuration
#define TM 4   // Each thread computes 4 rows
#define TN 2   // Each thread computes 2 columns  
#define BLOCK_SIZE 16
```

## 📊 Performance Results

> **Note:** These benchmarks were conducted on a Google Colab T4 GPU.

### Benchmark Configuration

- **Input dimensions:**  
  ```cuda
  int num_samples = 2048;
  int input_dim = 1024;
  int num_features = 5120;
- **Timing Configurations**
  ```cuda
  #define WARMUP_RUNS 3
  #define TIMING_RUNS 5

### Performance Improvements

| Operation | Naive (ms) | Optimized (ms) | Speedup |
|-----------|------------|----------------|---------|
| Forward Pass | 51.44 | 29.67 | **1.73×** |
| Backward Pass | 95.81 | 42.50 | **2.25×** |
| **Total Pipeline** | 147.25 | 72.17 | **2.04×** |


## 🛠️ Build & Run

There are 2 files:

1. **MatMulOptimizationKernels.cu**  
   A step-by-step implementation of matrix multiplication kernel optimizations. This file was developed as a learning exercise, following [this blog](https://siboehm.com/articles/22/CUDA-MMM). Each optimization—such as tiling, vectorization, and shared memory usage—was implemented incrementally to observe and measure performance improvements.
Here are results:
```
=== PERFORMANCE RESULTS ===--- Naive GPU Kernel ---Avg time: 46.252 ms
--- Optimized GPU Kernel ---Avg time: 189.697 <-- See below
--- Shared Memory GPU Kernel ---Avg time: 29.353 msShared vs Naive: 1.58xShared vs Optimized: 6.46x
--- Thread-Tiled GPU Kernel (TM=4, TN=2) ---Avg time: 24.813 msThreads per block: 8x4 = 32 (vs 16x16 = 256)Tiled vs Naive: 1.86xTiled vs Shared: 1.18x
--- Vectorized Shared Memory GPU Kernel ---Avg time: 14.071 msVectorized vs Naive: 3.29xVectorized vs Shared: 2.09xVectorized vs Thread-Tiled: 1.76xVectorized vs cuBLAS: 0.23x
--- cuBLAS ---Avg time: 3.169 mscuBLAS vs Naive: 14.59xcuBLAS vs Optimized: 59.86xcuBLAS vs Shared: 9.26x <-- See below
=== CORRECTNESS CHECK ===Max abs diff (Naive vs Optimized): 0.000000e+00Max abs diff (Naive vs Shared): 0.000000e+00Max abs diff (Naive vs Thread-Tiled): 0.000000e+00Max abs diff (Naive vs Vectorized): 0.000000e+00Max abs diff (Naive vs cuBLAS): 0.000000e+00
```
- Optimized GPU Kernel: *Attempted to transpose W matrix but only increased complexity.*
- cuBLAS: *As you can see, we are still far away from the heavily optimized cuBLAS function. *😢

2. **OptimizedFullyConnectedNN.cu**  
   A fully connected neural network layer that incorporates all the CUDA optimization techniques from the first file. This implementation includes both the forward and backward passes, and is optimized using shared memory, vectorized memory access, and thread-level parallelism.

🖥️ As I did not have access to a local GPU, all testing/developing conducted using **Google Colab**. If you're in a similar situation, you can explore the /colab folder, which contains ready-to-run Colab notebooks.
For users with access to a CUDA-capable GPU, the raw .cu files are located in the /cuda folder and can be compiled and executed locally.

### Future Optimization Opportunities
- **Half-precision (FP16)**: memory bandwidth improvement
- **Tensor Cores**: Leverage mixed-precision matrix operations
- **Multi-GPU**: Scale across multiple devices
- **Kernel Fusion**: Combine forward/backward passes
- **Warl-Level Optimization**: Imrpove Synchronization/Data sharing

## 📚 Learning Resources
- https://siboehm.com/articles/22/CUDA-MMM
- https://www.youtube.com/watch?v=86FAWCzIe_4&t=36117s

