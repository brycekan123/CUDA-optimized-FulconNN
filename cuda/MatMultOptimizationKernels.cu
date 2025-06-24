#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 16
#define RUNS 5
#define TM 4
#define TN 2
//Results are from using google colab T4 GPU. I also wrote in jupyter notebook if you want to run yourself on google colab
// int num_samples = 2048, input_dim = 1024, num_features = 5120;

// Kernel 1: This is basic/naive implementation of matrix multiplication in CUDA. 
// We uses a single thread to perform multiplication row-wise in X and column-wise in W. 
// Then accumulate the sum and write to a cell in the resulting matrix
// Result: 46ms
// Notes: going column-wise on W is not ideal as accessing the next row in W requires hopping num_samples memory blocks. 
//        This can be seen in W[i * num_features + col] where i is incremented, resulting in access to W to be inefficient. 
__global__ void matmul_kernel(const float* X, const float* W, float* Y,
                              int num_samples, int input_dim, int num_features) {
    //blockIdx = block index in grid
    // blockDim = total num threads in block
    // threadidx = thread index in block
    //ex: to get thread 54 in y dimension, block idx.y = 3, block dim = 16, threadidx = 6.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_samples && col < num_features) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += X[row * input_dim + i] * W[i * num_features + col];
        }
        Y[row * num_features + col] = sum;
    }
}
// Kernel 2: Accessing W in row-wise fashion
// To make access to W more efficient, I transposed W first in a seperate kernel
// Now, I can access both X and W in a row-wise fashion when doing matrix multiplication. 
// Result: 191ms
// !!! The overhead of transposing W was costly and actually inefficient. a logical attempt but not worth in this instance.

__global__ void transpose_kernel(const float* W, float* W_t, int input_dim, int num_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < input_dim && col < num_features) {
        W_t[col * input_dim + row] = W[row * num_features + col];
    }
}
__global__ void matmul_kernel_optimized(const float* X, const float* W_t, float* Y,
                                        int num_samples, int input_dim, int num_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_samples && col < num_features) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += X[row * input_dim + i] * W_t[col * input_dim + i];
        }
        Y[row * num_features + col] = sum;
    }
}

// Kernel 3:
// Now, I'm implementing shared memory/tiling. Previously, we're accessing all matrices in the global memory space.
// To reduce the number of times I access global memory, we take chunks or "tiles" of the matrices to shared memory, which is much faster access.
// Tile sizes in this case are 16x16. num of Threads remain the same! 
// Each thread is still responsible for a single matmult in resulting matrix
// Each thread holds the partial sum in the tiles.
// Result: 28ms
__global__ void matmul_kernel_shared(const float* X, const float* W, float* Y,
                                     int num_samples, int input_dim, int num_features) {
    // Shared memory for tiles
    __shared__ float X_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float W_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Thread and block indices
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Global indices for this thread
    int globalRow = blockRow * BLOCK_SIZE + threadRow;
    int globalCol = blockCol * BLOCK_SIZE + threadCol;

    // EACH SUM IS LOCAL TO THE THREAD
    float sum = 0.0f;

    // Loop over tiles along the K dimension (input_dim)
    int numTiles = (input_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // we iterate over all tiles. Threads are operating in parallel for every tile index INSIDE the for loop
    // We use same threads per iteration
    //the SAME thread iterates over tileidx. extracting 16x16 from global memory. and adding it to running sum per thread
    // In this case, there are 256 threads within this block to load the 16x16 tiles.

    // we also run blocks in parallel. so each block has 256 UNIQUE threads.
    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        // Calculate tile starting positions
        int X_tileCol = tileIdx * BLOCK_SIZE + threadCol;
        int W_tileRow = tileIdx * BLOCK_SIZE + threadRow;


        // Load X tile into shared memory
        // Each thread loads one element: X[globalRow][X_tileCol]
        // There are 256 threads. This loads the X_tile as 16x16
        //
        if (globalRow < num_samples && X_tileCol < input_dim) {
            X_tile[threadRow][threadCol] = X[globalRow * input_dim + X_tileCol];
        } else {
            X_tile[threadRow][threadCol] = 0.0f;
        }

        // Load W tile into shared memory
        // Each thread loads one element: W[W_tileRow][globalCol]
        if (W_tileRow < input_dim && globalCol < num_features) {
            W_tile[threadRow][threadCol] = W[W_tileRow * num_features + globalCol];
        } else {
            W_tile[threadRow][threadCol] = 0.0f;
        }

        // Synchronize to make sure tiles are loaded
        __syncthreads();

        //X_tile and W_tile get overwritten.

        // Running Sum! The sum's are saved individually PER THREAD.
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += X_tile[threadRow][k] * W_tile[k][threadCol];
        }
        __syncthreads();
    }

    // Write result to global memory
    if (globalRow < num_samples && globalCol < num_features) {
        Y[globalRow * num_features + globalCol] = sum;
    }
}

//Kernel 4: Increasing arithmatic indentensity by reducing number of threads being used.
// To further optimize, we reduce have threads compute several results in resulting matrix.TM and TN. 
// This is stored in threadresults.
// also reduced by caching rows of W .
// Result: 23ms
__global__ void matmul_kernel_threadreduce(const float* X, const float* W, float* Y,
                                        int num_samples, int input_dim, int num_features) {

    // Shared memory for tiles
    __shared__ float X_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float W_tile[BLOCK_SIZE][BLOCK_SIZE];

    // Thread and block indices
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // With thread tiling, we need fewer threads per block
    // Block dimensions are now (BLOCK_SIZE/TN, BLOCK_SIZE/TM)
    // Each thread handles TM x TN output elements

    // Starting global positions for this thread's output region
    int globalRowStart = blockRow * BLOCK_SIZE + threadRow * TM;
    int globalColStart = blockCol * BLOCK_SIZE + threadCol * TN;

    // Thread-local register cache for results
    // Each thread accumulates TM x TN results
    float threadResults[TM][TN];

    // Initialize results to zero
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            threadResults[m][n] = 0.0f;
        }
    }

    // Loop over tiles along the K dimension (input_dim)
    int numTiles = (input_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        // Calculate tile starting positions
        int X_tileCol = tileIdx * BLOCK_SIZE;
        int W_tileRow = tileIdx * BLOCK_SIZE;

        // Load X tile into shared memory
        // Each thread loads multiple elements due to fewer threads
        for (int m = 0; m < TM; m++) {
            for (int k = 0; k < TN; k++) {
                int loadRow = globalRowStart + m;
                int loadCol = X_tileCol + threadCol * TN + k;
                int smemRow = threadRow * TM + m;
                int smemCol = threadCol * TN + k;

                if (loadRow < num_samples && loadCol < input_dim &&
                    smemRow < BLOCK_SIZE && smemCol < BLOCK_SIZE) {
                    X_tile[smemRow][smemCol] = X[loadRow * input_dim + loadCol];
                } else if (smemRow < BLOCK_SIZE && smemCol < BLOCK_SIZE) {
                    X_tile[smemRow][smemCol] = 0.0f;
                }
            }
        }

        // Load W tile into shared memory
        // Each thread loads multiple elements due to fewer threads
        for (int k = 0; k < TM; k++) {
            for (int n = 0; n < TN; n++) {
                int loadRow = W_tileRow + threadRow * TM + k;
                int loadCol = globalColStart + n;
                int smemRow = threadRow * TM + k;
                int smemCol = threadCol * TN + n;

                if (loadRow < input_dim && loadCol < num_features &&
                    smemRow < BLOCK_SIZE && smemCol < BLOCK_SIZE) {
                    W_tile[smemRow][smemCol] = W[loadRow * num_features + loadCol];
                } else if (smemRow < BLOCK_SIZE && smemCol < BLOCK_SIZE) {
                    W_tile[smemRow][smemCol] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial results for this tile
        // Each thread computes TM x TN partial dot products
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // Cache W values for reuse across TM iterations
            float W_cache[TN];
            for (int n = 0; n < TN; n++) {
                W_cache[n] = W_tile[k][threadCol * TN + n];
            }

            // Compute TM x TN partial products
            for (int m = 0; m < TM; m++) {
                float X_val = X_tile[threadRow * TM + m][k];
                for (int n = 0; n < TN; n++) {
                    threadResults[m][n] += X_val * W_cache[n];
                }
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    // Each thread writes TM x TN output elements
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int outputRow = globalRowStart + m;
            int outputCol = globalColStart + n;

            if (outputRow < num_samples && outputCol < num_features) {
                Y[outputRow * num_features + outputCol] = threadResults[m][n];
            }
        }
    }
}

// Kernel 5:
// Building on top of thread reduction, now we implement vectorization
// Everything is the same except previously, we loaded from global memory to shared memory(tile) at 1 float per iteration.
// Since the locations of floats are next to each other, we can use float4 to get 4 floats at a time PER global mem access
// This allows for less global memorry access.
// The only thing that changed: Tile size is increased and XTile is loaded vectorized. Wtile is still loaded same way
// I also implemented #pragma unrolling for slightly more efficient access during for loops
// Result: 13ms

__global__ void matmul_vectorized(const float* X, const float* W, float* Y,
                                   int num_samples, int input_dim, int num_features) {

    __shared__ float X_tile[BLOCK_SIZE][BLOCK_SIZE * 4 + 1];  
    __shared__ float W_tile[BLOCK_SIZE * 4][BLOCK_SIZE + 1];  

    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    const int globalRowStart = blockIdx.y * BLOCK_SIZE + threadRow * TM;
    const int globalColStart = blockIdx.x * BLOCK_SIZE + threadCol * TN;

    float threadResults[TM][TN];

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            threadResults[m][n] = 0.0f;
        }
    }

    const int numTiles = (input_dim + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        const int X_tileCol = tileIdx * BLOCK_SIZE * 4;
        const int W_tileRow = tileIdx * BLOCK_SIZE * 4;

        // Load X_tile with your optimized approach
        #pragma unroll
        for (int m = 0; m < TM; m++) {
            const int globalRow = globalRowStart + m;
            const int colsPerThread = (BLOCK_SIZE * 4) / (BLOCK_SIZE / TN);

            #pragma unroll
            for (int i = 0; i < colsPerThread; i += 4) {
                const int tileCol = threadCol * colsPerThread + i;

                if (tileCol + 3 < BLOCK_SIZE * 4) {
                    const int globalCol = X_tileCol + tileCol;

                    if (globalRow < num_samples && globalCol + 3 < input_dim) {
                        // Vectorized load with proper alignment
                        const float4 vecX = __ldg(reinterpret_cast<const float4*>(&X[globalRow * input_dim + globalCol]));
                        X_tile[threadRow * TM + m][tileCol + 0] = vecX.x;
                        X_tile[threadRow * TM + m][tileCol + 1] = vecX.y;
                        X_tile[threadRow * TM + m][tileCol + 2] = vecX.z;
                        X_tile[threadRow * TM + m][tileCol + 3] = vecX.w;
                    } else {
                        //if total columns are not divisible by 4
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const int col = globalCol + j;
                            X_tile[threadRow * TM + m][tileCol + j] =
                                (globalRow < num_samples && col < input_dim) ?
                                __ldg(&X[globalRow * input_dim + col]) : 0.0f;
                        }
                    }
                }
            }
        }
        
        #pragma unroll
        for (int k = 0; k < TM; k++) {
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int loadRow = W_tileRow + threadRow * TM + k;
                int loadCol = globalColStart + n;
                int smemRow = threadRow * TM + k;
                int smemCol = threadCol * TN + n;

                if (loadRow < input_dim && loadCol < num_features &&
                    smemRow < BLOCK_SIZE * 4 && smemCol < BLOCK_SIZE) {
                    W_tile[smemRow][smemCol] = W[loadRow * num_features + loadCol];
                } else if (smemRow < BLOCK_SIZE * 4 && smemCol < BLOCK_SIZE) {
                    W_tile[smemRow][smemCol] = 0.0f;
                }
            }
        }
    
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE * 4; k++) {
            float W_cache[TN];
            for (int n = 0; n < TN; n++) {
                W_cache[n] = W_tile[k][threadCol * TN + n];
            }
            // Compute TM x TN partial products
            for (int m = 0; m < TM; m++) {
                float X_val = X_tile[threadRow * TM + m][k];
                for (int n = 0; n < TN; n++) {
                    threadResults[m][n] += X_val * W_cache[n];
                }
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    // Each thread writes TM x TN output elements
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int outputRow = globalRowStart + m;
            int outputCol = globalColStart + n;
            if (outputRow < num_samples && outputCol < num_features) {
                Y[outputRow * num_features + outputCol] = threadResults[m][n];
            }
        }
    }
}
// cuBLAS result: 3.108ms
int main() {
    int num_samples = 2048, input_dim = 1024, num_features = 5120;
    size_t size_X = num_samples * input_dim * sizeof(float);
    size_t size_W = input_dim * num_features * sizeof(float);
    size_t size_Y = num_samples * num_features * sizeof(float);

    float *h_X = (float*)malloc(size_X);
    float *h_W = (float*)malloc(size_W);
    float *h_W_t = (float*)malloc(size_W);
    float *h_Y_gpu_naive = (float*)malloc(size_Y);
    float *h_Y_gpu_opt = (float*)malloc(size_Y);
    float *h_Y_gpu_shared = (float*)malloc(size_Y);
    float *h_Y_gpu_tiled = (float*)malloc(size_Y);
    float* h_Y_gpu_vectorized = (float*)malloc(size_Y);

    float *h_Y_cublas = (float*)malloc(size_Y);

    for (int i = 0; i < num_samples * input_dim; i++) {
        h_X[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < input_dim * num_features; i++) {
        h_W[i] = (float)rand() / RAND_MAX;
    }

    float *d_X, *d_W, *d_W_t,*d_Y_vec, *d_Y;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_W_t, size_W);
    cudaMalloc(&d_Y, size_Y);
    cudaMalloc(&d_Y_vec, size_Y);


    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Thread-tiled kernel configuration
    dim3 threadsPerBlock_tiled(BLOCK_SIZE/TN, BLOCK_SIZE/TM);
    dim3 numBlocks_tiled((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                         (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);


    // Set up grid and block dimensions for vectorized kernel
    dim3 threadsPerBlock_vectorized(BLOCK_SIZE / TN, BLOCK_SIZE / TM);
    dim3 numBlocks_vectorized((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Transpose W on GPU
    dim3 threads_t(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_t((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (input_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);

    transpose_kernel<<<blocks_t, threads_t>>>(d_W, d_W_t, input_dim, num_features);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup kernels
    matmul_kernel<<<blocks, threads>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features);
    cudaDeviceSynchronize();

    matmul_kernel_optimized<<<blocks, threads>>>(d_X, d_W_t, d_Y, num_samples, input_dim, num_features);
    cudaDeviceSynchronize();

    matmul_kernel_shared<<<blocks, threads>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features);
    cudaDeviceSynchronize();

    matmul_kernel_threadreduce<<<numBlocks_tiled, threadsPerBlock_tiled>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features);
    cudaDeviceSynchronize();

    matmul_vectorized<<<numBlocks_vectorized, threadsPerBlock_vectorized>>>(
        d_X, d_W, d_Y, num_samples, input_dim, num_features);
    cudaDeviceSynchronize();

    // cuBLAS warmup
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                num_features, num_samples, input_dim,
                &alpha, d_W, num_features, d_X, input_dim, &beta, d_Y, num_features);
    cudaDeviceSynchronize();

    // Naive GPU kernel timing
    float total_naive_gpu_time = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_kernel<<<blocks, threads>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float this_time;
        cudaEventElapsedTime(&this_time, start, stop);
        total_naive_gpu_time += this_time;
    }
    float avg_naive_gpu_time = total_naive_gpu_time / RUNS;
    cudaMemcpy(h_Y_gpu_naive, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // Optimized GPU kernel timing
    float total_opt_gpu_time = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_kernel_optimized<<<blocks, threads>>>(d_X, d_W_t, d_Y, num_samples, input_dim, num_features);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float this_time;
        cudaEventElapsedTime(&this_time, start, stop);
        total_opt_gpu_time += this_time;
    }
    float avg_opt_gpu_time = total_opt_gpu_time / RUNS;
    cudaMemcpy(h_Y_gpu_opt, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // Shared memory GPU kernel timing
    float total_shared_time = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_kernel_shared<<<blocks, threads>>>(d_X, d_W, d_Y,
                                                  num_samples, input_dim, num_features);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float this_time;
        cudaEventElapsedTime(&this_time, start, stop);
        total_shared_time += this_time;
    }
    float avg_shared_time = total_shared_time / RUNS;
    cudaMemcpy(h_Y_gpu_shared, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // Thread-tiled GPU kernel timing
    float total_tiled_time = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_kernel_threadreduce<<<numBlocks_tiled, threadsPerBlock_tiled>>>(d_X, d_W, d_Y,
                                                                       num_samples, input_dim, num_features);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float this_time;
        cudaEventElapsedTime(&this_time, start, stop);
        total_tiled_time += this_time;
    }
    float avg_tiled_time = total_tiled_time / RUNS;
    cudaMemcpy(h_Y_gpu_tiled, d_Y, size_Y, cudaMemcpyDeviceToHost);



    // Timing runs for vectorized kernel
    float total_vectorized_time = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        matmul_vectorized<<<numBlocks_vectorized, threadsPerBlock_vectorized>>>(
            d_X, d_W, d_Y_vec, num_samples, input_dim, num_features);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float this_time;
        cudaEventElapsedTime(&this_time, start, stop);
        total_vectorized_time += this_time;
    }
    float avg_vectorized_time = total_vectorized_time / RUNS;
    cudaMemcpy(h_Y_gpu_vectorized, d_Y, size_Y, cudaMemcpyDeviceToHost);


    float total_cublas_time = 0.0f;
    for (int i = 0; i < RUNS; ++i) {
        cudaEventRecord(start);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    num_features, num_samples, input_dim,
                    &alpha, d_W, num_features, d_X, input_dim, &beta, d_Y, num_features);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float this_time;
        cudaEventElapsedTime(&this_time, start, stop);
        total_cublas_time += this_time;
    }
    float avg_cublas_time = total_cublas_time / RUNS;
    cudaMemcpy(h_Y_cublas, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // Compare results for correctness
    float max_diff_naive_opt = 0.0f;
    float max_diff_naive_shared = 0.0f;
    float max_diff_naive_tiled = 0.0f;
    float max_diff_naive_cublas = 0.0f;
    float max_diff_naive_vectorized = 0.0f;

    for (int i = 0; i < num_samples * num_features; ++i) {
        float diff_opt = fabs(h_Y_gpu_naive[i] - h_Y_gpu_opt[i]);
        float diff_shared = fabs(h_Y_gpu_naive[i] - h_Y_gpu_shared[i]);
        float diff_tiled = fabs(h_Y_gpu_naive[i] - h_Y_gpu_tiled[i]);
        float diff_cublas = fabs(h_Y_gpu_naive[i] - h_Y_cublas[i]);
        float diff_vectorized = fabs(h_Y_gpu_naive[i] - h_Y_gpu_vectorized[i]);

        if (diff_opt > max_diff_naive_opt) max_diff_naive_opt = diff_opt;
        if (diff_shared > max_diff_naive_shared) max_diff_naive_shared = diff_shared;
        if (diff_tiled > max_diff_naive_tiled) max_diff_naive_tiled = diff_tiled;
        if (diff_cublas > max_diff_naive_cublas) max_diff_naive_cublas = diff_cublas;
        if (diff_vectorized > max_diff_naive_vectorized) max_diff_naive_vectorized = diff_vectorized;
    }

    // Print results
    printf("=== PERFORMANCE RESULTS ===");
    printf("--- Naive GPU Kernel ---");
    printf("Avg time: %.3f ms", avg_naive_gpu_time);
    puts("");
    printf("--- Optimized GPU Kernel ---");
    printf("Avg time: %.3f ms", avg_opt_gpu_time);
    printf("Naive vs Optimized: %.2fx", avg_opt_gpu_time / avg_naive_gpu_time);
    puts("");
    printf("--- Shared Memory GPU Kernel ---");
    printf("Avg time: %.3f ms", avg_shared_time);
    printf("Shared vs Naive: %.2fx", avg_naive_gpu_time / avg_shared_time);
    printf("Shared vs Optimized: %.2fx", avg_opt_gpu_time / avg_shared_time);
    puts("");
    printf("--- Thread-Tiled GPU Kernel (TM=%d, TN=%d) ---", TM, TN);
    printf("Avg time: %.3f ms", avg_tiled_time);
    printf("Threads per block: %dx%d = %d (vs %dx%d = %d)",
           threadsPerBlock_tiled.x, threadsPerBlock_tiled.y,
           threadsPerBlock_tiled.x * threadsPerBlock_tiled.y,
           threads.x, threads.y, threads.x * threads.y);
    printf("Tiled vs Naive: %.2fx", avg_naive_gpu_time / avg_tiled_time);
    printf("Tiled vs Shared: %.2fx", avg_shared_time / avg_tiled_time);
    puts("");

    printf("--- Vectorized Shared Memory GPU Kernel ---");
    printf("Avg time: %.3f ms", avg_vectorized_time);
    printf("Vectorized vs Naive: %.2fx", avg_naive_gpu_time / avg_vectorized_time);
    printf("Vectorized vs Shared: %.2fx", avg_shared_time / avg_vectorized_time);
    printf("Vectorized vs Thread-Tiled: %.2fx", avg_tiled_time / avg_vectorized_time);
    printf("Vectorized vs cuBLAS: %.2fx", avg_cublas_time / avg_vectorized_time);
    puts("");

    printf("--- cuBLAS ---");
    printf("Avg time: %.3f ms", avg_cublas_time);
    printf("cuBLAS vs Naive: %.2fx", avg_naive_gpu_time / avg_cublas_time);
    printf("cuBLAS vs Optimized: %.2fx", avg_opt_gpu_time / avg_cublas_time);
    printf("cuBLAS vs Shared: %.2fx", avg_shared_time / avg_cublas_time);
    puts("");

    printf("=== CORRECTNESS CHECK ===");
    printf("Max abs diff (Naive vs Optimized): %e", max_diff_naive_opt);
    printf("Max abs diff (Naive vs Shared): %e", max_diff_naive_shared);
    printf("Max abs diff (Naive vs Thread-Tiled): %e", max_diff_naive_tiled);
    printf("Max abs diff (Naive vs Vectorized): %e", max_diff_naive_vectorized);
    printf("Max abs diff (Naive vs cuBLAS): %e", max_diff_naive_cublas);

    // Cleanup
    cublasDestroy(cublas_handle);
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_W_t);
    cudaFree(d_Y);
    free(h_X);
    free(h_W);
    free(h_W_t);
    free(h_Y_gpu_naive);
    free(h_Y_gpu_opt);
    free(h_Y_gpu_shared);
    free(h_Y_cublas);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
