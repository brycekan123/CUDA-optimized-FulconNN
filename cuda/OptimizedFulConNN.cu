#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>

#define TM 4  
#define TN 2  
#define BLOCK_SIZE 16
#define WARMUP_RUNS 3  
#define TIMING_RUNS 5  

// Using the tools I learned in MatMult, I decided to implement them in a forward and backward pass.
// First, I implemented the naive version of forward and backward pass.
// Then, I used tiling and vectorization to optimize them, resulting in ~2x speed up


//Naive version of forward pass
//I also added bias and a sigmoid activation function into the forward pass
__global__ void forward_naive(const float* X, const float* W, float* Y,
                              int num_samples, int input_dim, int num_features, const float* bias) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_samples && col < num_features) {
        float sum = 0.0f;
        for (int i = 0; i < input_dim; ++i) {
            sum += X[row * input_dim + i] * W[i * num_features + col];
        }
        sum += bias[col];
        sum = 1.0f / (1.0f + expf(-sum));  
        Y[row * num_features + col] = sum;
    }
}


// Optimized version of forward pass
// First, I wanted to implement vectorization and tiling. 
__global__ void forward_vectorized(const float* X, const float* W, float* Y,
                                   int num_samples, int input_dim, int num_features, const float* bias) {

    // Shared memory for tiles
    __shared__ float X_tile[BLOCK_SIZE][BLOCK_SIZE * 4];
    __shared__ float W_tile[BLOCK_SIZE * 4][BLOCK_SIZE];

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int globalRowStart = blockIdx.y * BLOCK_SIZE + threadRow * TM;
    int globalColStart = blockIdx.x * BLOCK_SIZE + threadCol * TN;
    
    //threads hold 4x2 results = 8
    float threadResults[TM][TN] = {0};

    int numTiles = (input_dim + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        int tileColStart = tileIdx * BLOCK_SIZE * 4;
        int tileRowStart = tileIdx * BLOCK_SIZE * 4;

        // Load X_tile: Each thread loads multiple elements
        // X_tile[BLOCK_SIZE][BLOCK_SIZE * 4] maps to X[num_samples][input_dim]
        for (int m = 0; m < TM; m++) {
            int globalRow = globalRowStart + m;

            // Each thread loads 4 consecutive elements per iteration
            for (int loadIdx = 0; loadIdx < (BLOCK_SIZE * 4) / (BLOCK_SIZE / TN); loadIdx += 4) {
                int tileCol = threadCol * ((BLOCK_SIZE * 4) / (BLOCK_SIZE / TN)) + loadIdx;
                int globalCol = tileColStart + tileCol;

                if (globalRow < num_samples && globalCol + 3 < input_dim && tileCol + 3 < BLOCK_SIZE * 4) {
                    // Vectorized load. Take 4 floats at a time if globalcol and titlecol are divisible by 4
                    float4 vecX = *reinterpret_cast<const float4*>(&X[globalRow * input_dim + globalCol]);
                    X_tile[threadRow * TM + m][tileCol + 0] = vecX.x;
                    X_tile[threadRow * TM + m][tileCol + 1] = vecX.y;
                    X_tile[threadRow * TM + m][tileCol + 2] = vecX.z;
                    X_tile[threadRow * TM + m][tileCol + 3] = vecX.w;
                } else {
                    // If not divisible by 4
                    for (int j = 0; j < 4 && tileCol + j < BLOCK_SIZE * 4; j++) {
                        int col = globalCol + j;
                        X_tile[threadRow * TM + m][tileCol + j] =
                            (globalRow < num_samples && col < input_dim) ?
                            X[globalRow * input_dim + col] : 0.0f;
                    }
                }
            }
        }

        // Load W_tile: Each thread loads elements for its output columns
        // W_tile can't be loaded in vectorized form since I go column-wise. 
        // I tried to implement row-wise by ended up with slower inference time.
        for (int loadRow = 0; loadRow < (BLOCK_SIZE * 4) / (BLOCK_SIZE / TM); loadRow++) {
            int tileRow = threadRow * ((BLOCK_SIZE * 4) / (BLOCK_SIZE / TM)) + loadRow;
            int globalRow = tileRowStart + tileRow;

            for (int n = 0; n < TN; n++) {
                int globalCol = globalColStart + n;

                if (globalRow < input_dim && globalCol < num_features && tileRow < BLOCK_SIZE * 4) {
                    W_tile[tileRow][threadCol * TN + n] = W[globalRow * num_features + globalCol];
                } else if (tileRow < BLOCK_SIZE * 4) {
                    W_tile[tileRow][threadCol * TN + n] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < BLOCK_SIZE * 4; k++) {
            for (int m = 0; m < TM; m++) {
                float x_val = X_tile[threadRow * TM + m][k];
                for (int n = 0; n < TN; n++) {
                    threadResults[m][n] += x_val * W_tile[k][threadCol * TN + n];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory with bias and activation
    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
            int row = globalRowStart + m;
            int col = globalColStart + n;

            if (row < num_samples && col < num_features) {
                float result = threadResults[m][n] + bias[col];  // Add bias
                result = 1.0f / (1.0f + expf(-result));  // Sigmoid activation
                Y[row * num_features + col] = result;
            }
        }
    }
}

// Naive implementation of backward pass
__global__ void backward_naive(
    const float* X, const float* Y_pred, const float* Y_true,
    float* dW,
    int num_samples, int input_dim, int num_features)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // input dimension index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // feature dimension index

    if (row >= input_dim || col >= num_features) return;

    float grad_w = 0.0f;

    // Accumulate gradient across all samples
    for (int s = 0; s < num_samples; ++s) {
        float y_pred = Y_pred[s * num_features + col];
        float y_true = Y_true[s * num_features + col];
        // gradient of loss function!(y_pred - y_true)
        // sigmoid derivative=(y_pred * (1.0f - y_pred);))
        float dL_dZ = (y_pred - y_true) * y_pred * (1.0f - y_pred);

        // Accumulate gradient: dL/dW = X^T * dL/dZ
        grad_w += X[s * input_dim + row] * dL_dZ;
    }
    //full gradient matrix
    dW[row * num_features + col] = grad_w;
}


// naive version of backward bias
// gradient of bias is the sum of dl_dz. 
__global__ void backward_bias_naive(
    const float* Y_pred, const float* Y_true,
    float* db,
    int num_samples, int num_features)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // feature dimension index

    if (j >= num_features) return;

    float grad_b = 0.0f;

    // Accumulate gradient across all samples
    for (int s = 0; s < num_samples; ++s) {
        float y_pred = Y_pred[s * num_features + j];
        float y_true = Y_true[s * num_features + j];

        // Compute gradient of loss w.r.t. pre-activation (sigmoid derivative)
        float dL_dZ = (y_pred - y_true) * y_pred * (1.0f - y_pred);

        // Accumulate bias gradient: dL/db = sum(dL/dZ)
        grad_b += dL_dZ;
    }

    db[j] = grad_b;
}


// I ran into issues when trying to implement vectorization on the backwards pass
// I resulted in just implementing tiling and increasing num of results the threads can hold
__global__ void backward_optimized(
    const float* X, const float* Y_pred, const float* Y_true,
    float* dW,
    int num_samples, int input_dim, int num_features)
{
    __shared__ float X_tile[BLOCK_SIZE * 4][BLOCK_SIZE];
    __shared__ float dL_dZ_tile[BLOCK_SIZE][BLOCK_SIZE];

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int globalRowStart = blockIdx.y * BLOCK_SIZE + threadRow * TM;
    int globalColStart = blockIdx.x * BLOCK_SIZE + threadCol * TN;

    float threadResults[TM][TN] = {0};

    int numTiles = (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma unroll
    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        int sampleTileStart = tileIdx * BLOCK_SIZE;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int inputRow = globalRowStart + i;
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int sampleCol = sampleTileStart + threadCol * TN + n;

                if (inputRow < input_dim && sampleCol < num_samples) {
                    X_tile[threadRow * 4 + i][threadCol * TN + n] = X[sampleCol * input_dim + inputRow];
                } else {
                    X_tile[threadRow * 4 + i][threadCol * TN + n] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < TM; m++) {
            int sampleRow = sampleTileStart + threadRow * TM + m;
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                int featureCol = globalColStart + n;

                if (sampleRow < num_samples && featureCol < num_features) {
                    // Use precomputed forward pass results
                    float y_pred = Y_pred[sampleRow * num_features + featureCol];
                    float y_true = Y_true[sampleRow * num_features + featureCol];

                    // Compute gradient of loss w.r.t. z
                    dL_dZ_tile[threadRow * TM + m][threadCol * TN + n] =
                        (y_pred - y_true) * y_pred * (1.0f - y_pred);
                } else {
                    dL_dZ_tile[threadRow * TM + m][threadCol * TN + n] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute partial gradient: X^T * dL_dZ
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            // Cache dL_dZ values for this k
            float dL_dZ_cache[TN];
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                dL_dZ_cache[n] = dL_dZ_tile[k][threadCol * TN + n];
            }

            #pragma unroll
            for (int m = 0; m < TM; m++) {
                float x_val = X_tile[threadRow * TM + m][k];
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    threadResults[m][n] += x_val * dL_dZ_cache[n];
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int inputRow = globalRowStart + m;
            int featureCol = globalColStart + n;

            if (inputRow < input_dim && featureCol < num_features) {
                dW[inputRow * num_features + featureCol] = threadResults[m][n];
            }
        }
    }
}

//I was able to implement vectorization of the bias part of the backwards pass
__global__ void backward_bias_vectorized(
    const float* Y_pred, const float* Y_true,
    float* db,
    int num_samples, int num_features)
{
    const int globalColStart = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalColStart * 4 >= num_features) return;
    
    float threadResults[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Process 4 samples at a time
    int s = 0;
    for (; s + 3 < num_samples; s += 4) {
        #pragma unroll
        for (int sample_offset = 0; sample_offset < 4; sample_offset++) {
            const int base_idx = (s + sample_offset) * num_features + globalColStart * 4;
            
            if (globalColStart * 4 + 3 < num_features) {
                // Vectorized load with proper alignment
                const float4 vecPred = __ldg(reinterpret_cast<const float4*>(&Y_pred[base_idx]));
                const float4 vecTrue = __ldg(reinterpret_cast<const float4*>(&Y_true[base_idx]));
                
                threadResults[0] += (vecPred.x - vecTrue.x) * vecPred.x * (1.0f - vecPred.x);
                threadResults[1] += (vecPred.y - vecTrue.y) * vecPred.y * (1.0f - vecPred.y);
                threadResults[2] += (vecPred.z - vecTrue.z) * vecPred.z * (1.0f - vecPred.z);
                threadResults[3] += (vecPred.w - vecTrue.w) * vecPred.w * (1.0f - vecPred.w);
            } else {
                // Handle boundary case
                #pragma unroll
                for (int i = 0; i < 4 && globalColStart * 4 + i < num_features; i++) {
                    const float y_pred = __ldg(&Y_pred[base_idx + i]);
                    const float y_true = __ldg(&Y_true[base_idx + i]);
                    threadResults[i] += (y_pred - y_true) * y_pred * (1.0f - y_pred);
                }
            }
        }
    }
    
    // Write back results
    if (globalColStart * 4 + 3 < num_features) {
        const float4 result_vec = {threadResults[0], threadResults[1], threadResults[2], threadResults[3]};
        *reinterpret_cast<float4*>(&db[globalColStart * 4]) = result_vec;
    } else {
        #pragma unroll
        for (int i = 0; i < 4 && globalColStart * 4 + i < num_features; i++) {
            db[globalColStart * 4 + i] = threadResults[i];
        }
    }
}

int main() {
    int num_samples = 2048, input_dim = 1024, num_features = 5120;

    size_t size_X = num_samples * input_dim * sizeof(float);
    size_t size_W = input_dim * num_features * sizeof(float);
    size_t size_Y = num_samples * num_features * sizeof(float);
    size_t size_B = num_features * sizeof(float);

    // Host memory allocation
    float *h_X = (float*)malloc(size_X);
    float *h_W = (float*)malloc(size_W);
    float *h_Y_true = (float*)malloc(size_Y);
    float *h_Y_pred_naive = (float*)malloc(size_Y);
    float *h_Y_pred_vectorized = (float*)malloc(size_Y);
    float *h_B = (float*)malloc(size_B);
    float *h_dW_naive = (float*)malloc(size_W);
    float *h_db_naive = (float*)malloc(size_B);
    float *h_dW_vectorized = (float*)malloc(size_W);
    float *h_db_vectorized = (float*)malloc(size_B);

    // Initialize data with random values
    srand(42);
    for (int i = 0; i < num_samples * input_dim; i++) {
        h_X[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    for (int i = 0; i < input_dim * num_features; i++) {
        h_W[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    for (int i = 0; i < num_features; i++) {
        h_B[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
    for (int i = 0; i < num_samples * num_features; i++) {
        h_Y_true[i] = (float)rand() / RAND_MAX;
    }

    // Device memory allocation
    float *d_X, *d_W, *d_Y, *d_B, *d_Y_true, *d_dW, *d_db;
    cudaMalloc(&d_X, size_X);
    cudaMalloc(&d_W, size_W);
    cudaMalloc(&d_Y, size_Y);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_Y_true, size_Y);
    cudaMalloc(&d_dW, size_W);
    cudaMalloc(&d_db, size_B);

    // Copy data to device
    cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, size_W, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_true, h_Y_true, size_Y, cudaMemcpyHostToDevice);

    // Grid and block dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks_forward((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blocks_backward((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                        (input_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blocks_bias((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 threads_bias(BLOCK_SIZE, 1);

    // Vectorized kernel dimensions
    dim3 threadsPerBlock_vec(BLOCK_SIZE / TN, BLOCK_SIZE / TM);
    dim3 blocksPerGrid_vec((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                          (num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE);

    dim3 threadsPerBlock_back(BLOCK_SIZE / TN, BLOCK_SIZE / TM);
    dim3 blocksPerGrid_back_weights((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                   (input_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 blocksPerGrid_back_bias((num_features + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
    dim3 threadsPerBlock_bias(BLOCK_SIZE / TN, 1);

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("=== WARMUP PHASE ===");
    // WARMUP RUNS - Don't measure these
    for (int run = 0; run < WARMUP_RUNS; ++run) {
        forward_naive<<<blocks_forward, threads>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features, d_B);
        forward_vectorized<<<blocksPerGrid_vec, threadsPerBlock_vec>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features, d_B);
        backward_naive<<<blocks_backward, threads>>>(d_X, d_Y, d_Y_true, d_dW,
                                                      num_samples, input_dim, num_features);
        backward_bias_naive<<<blocks_bias, threads_bias>>>(d_Y, d_Y_true, d_db,
                                                           num_samples, num_features);
        backward_optimized<<<blocksPerGrid_back_weights, threadsPerBlock_back>>>(d_X, d_Y, d_Y_true, d_dW,
                                                                                 num_samples, input_dim, num_features);
        backward_bias_vectorized<<<blocksPerGrid_back_bias, threadsPerBlock_bias>>>(d_Y, d_Y_true, d_db,
                                                                                  num_samples, num_features);
        cudaDeviceSynchronize(); 
    }
    printf("Warmup completed (%d runs)", WARMUP_RUNS);

    printf("=== TIMING PHASE ===");

    // TIMED FORWARD PASS - Naive version
    float total_forward_naive = 0.0f;
    for (int run = 0; run < TIMING_RUNS; ++run) {
        cudaEventRecord(start);
        forward_naive<<<blocks_forward, threads>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features, d_B);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_forward_naive += ms;
    }
    // Copy naive forward results back to host
    cudaMemcpy(h_Y_pred_naive, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // TIMED FORWARD PASS - Vectorized version
    float total_forward_vectorized = 0.0f;
    for (int run = 0; run < TIMING_RUNS; ++run) {
        cudaEventRecord(start);
        forward_vectorized<<<blocksPerGrid_vec, threadsPerBlock_vec>>>(d_X, d_W, d_Y, num_samples, input_dim, num_features, d_B);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_forward_vectorized += ms;
    }
    // Copy vectorized forward results back to host
    cudaMemcpy(h_Y_pred_vectorized, d_Y, size_Y, cudaMemcpyDeviceToHost);

    // TIMED BACKWARD PASS (using naive version)
    float total_backward = 0.0f;
    for (int run = 0; run < TIMING_RUNS; ++run) {

        cudaEventRecord(start);
        backward_naive<<<blocks_backward, threads>>>(d_X, d_Y, d_Y_true, d_dW,
                                                      num_samples, input_dim, num_features);
        backward_bias_naive<<<blocks_bias, threads_bias>>>(d_Y, d_Y_true, d_db,
                                                           num_samples, num_features);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_backward += ms;
    }

    // Copy backward results back to host
    cudaMemcpy(h_dW_naive, d_dW, size_W, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db_naive, d_db, size_B, cudaMemcpyDeviceToHost);

    // TIMED BACKWARD PASS (using vectorized version with precomputed Y_pred)
    float total_backward_optimized = 0.0f;
    for (int run = 0; run < TIMING_RUNS; ++run) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        // Use the precomputed Y_pred from the last forward pass
        backward_optimized<<<blocksPerGrid_back_weights, threadsPerBlock_back>>>(
            d_X, d_Y, d_Y_true, d_dW, num_samples, input_dim, num_features);
        backward_bias_vectorized<<<blocksPerGrid_back_bias, threadsPerBlock_bias>>>(
            d_Y, d_Y_true, d_db, num_samples, num_features);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_backward_optimized += ms;
    }

    cudaMemcpy(h_dW_vectorized, d_dW, size_W, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_db_vectorized, d_db, size_B, cudaMemcpyDeviceToHost);

    // VERIFICATION: Compare forward pass results
    printf("=== FORWARD PASS VERIFICATION ===");
    float max_forward_diff = 0.0f;
    float avg_forward_diff = 0.0f;
    for (int i = 0; i < num_samples * num_features; i++) {
        float diff = fabsf(h_Y_pred_naive[i] - h_Y_pred_vectorized[i]);
        if (diff > max_forward_diff) max_forward_diff = diff;
        avg_forward_diff += diff;
    }
    avg_forward_diff /= (num_samples * num_features);
    printf("Forward pass - Max difference: %.2e, Avg difference: %.2e", max_forward_diff, avg_forward_diff);

    printf("=== BACKWARD PASS VERIFICATION ===");
    float max_weight_diff = 0.0f;
    float avg_weight_diff = 0.0f;
    for (int i = 0; i < input_dim * num_features; i++) {
        float diff = fabsf(h_dW_naive[i] - h_dW_vectorized[i]);
        if (diff > max_weight_diff) max_weight_diff = diff;
        avg_weight_diff += diff;
    }
    avg_weight_diff /= (input_dim * num_features);
    printf("Weight gradients - Max difference: %.2e, Avg difference: %.2e", max_weight_diff, avg_weight_diff);

    float max_bias_diff = 0.0f;
    float avg_bias_diff = 0.0f;
    for (int i = 0; i < num_features; i++) {
        float diff = fabsf(h_db_naive[i] - h_db_vectorized[i]);
        if (diff > max_bias_diff) max_bias_diff = diff;
        avg_bias_diff += diff;
    }
    avg_bias_diff /= num_features;
    printf("Bias gradients - Max difference: %.2e, Avg difference: %.2e", max_bias_diff, avg_bias_diff);

    // Print results
    printf("=== FORWARD PASS SAMPLE ===");
    printf("First 5 predictions (Naive vs Vectorized):");
    for (int i = 0; i < 5; ++i) {
        printf("Y_pred[%d]: Naive=%.6f, Vectorized=%.6f, Y_true=%.6f",
               i, h_Y_pred_naive[i], h_Y_pred_vectorized[i], h_Y_true[i]);
    }

    printf("=== BACKWARD GRADIENT SAMPLES ==");
    printf("First 5 weight gradients (Naive vs Vectorized):");
    for (int i = 0; i < 5; ++i) {
        printf("dW[%d]: Naive=%.6f, Vectorized=%.6f", i, h_dW_naive[i], h_dW_vectorized[i]);
    }
    printf("First 5 bias gradients (Naive vs Vectorized):");
    for (int i = 0; i < 5; ++i) {
        printf("db[%d]: Naive=%.6f, Vectorized=%.6f", i, h_db_naive[i], h_db_vectorized[i]);
    }

    printf("=== PERFORMANCE RESULTS ===");
    printf("Forward pass (naive) avg time:      %.3f ms (over %d runs)",
           total_forward_naive / TIMING_RUNS, TIMING_RUNS);
    printf("Forward pass (vectorized) avg time: %.3f ms (over %d runs)",
           total_forward_vectorized / TIMING_RUNS, TIMING_RUNS);
    printf("Forward speedup (naive -> vectorized): %.2fx",
           total_forward_naive / total_forward_vectorized);
    printf("Backward pass naive avg time:       %.3f ms (over %d runs)",
           total_backward / TIMING_RUNS, TIMING_RUNS);
    printf("Backward pass vectorized avg time:  %.3f ms (over %d runs)",
           total_backward_optimized / TIMING_RUNS, TIMING_RUNS);
    printf("Backward speedup (naive -> vectorized): %.2fx",
           total_backward / total_backward_optimized);
    printf("total speedup (naive -> vectorized): %.2fx",
           (total_backward+total_forward_naive) / (total_forward_vectorized+total_backward_optimized));


    // Compute simple loss for verification (using naive results)
    float loss = 0.0f;
    for (int i = 0; i < num_samples * num_features; ++i) {
        float diff = h_Y_pred_naive[i] - h_Y_true[i];
        loss += diff * diff;
    }
    loss /= (num_samples * num_features);
    printf("Mean Squared Error:                 %.6f", loss);

    // Cleanup
    cudaFree(d_X); cudaFree(d_W); cudaFree(d_Y); cudaFree(d_B);
    cudaFree(d_Y_true); cudaFree(d_dW); cudaFree(d_db);
    free(h_X); free(h_W); free(h_Y_pred_naive); free(h_Y_pred_vectorized);
    free(h_Y_true); free(h_B); free(h_dW_naive); free(h_db_naive);
    free(h_dW_vectorized); free(h_db_vectorized);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}