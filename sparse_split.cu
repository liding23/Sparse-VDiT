#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

// Kernel to split matrix into sparse and dense matrices
__global__ void sparse_column_split(const half* input, half* dense_output, half* sparse_output,
                                    int N, int d, int threshold) {  // threshold is an integer
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < d) {
        // Count non-zero elements in the column
        int non_zero_count = 0;
        for (int row = 0; row < N; row++) {
            // Convert half to float, then to int to compare as integers
            if (static_cast<int>(__half2float(input[row * d + col])) != 0) {  // Compare the integer value of the half precision
                non_zero_count++;
            }
        }

        // Compare the number of non-zero elements in the column with the threshold
        if (non_zero_count < threshold) {
            // Sparse column: place in sparse matrix, rest set to zero in dense matrix
            for (int row = 0; row < N; row++) {
                sparse_output[row * d + col] = input[row * d + col];
                dense_output[row * d + col] = __half(0.0f);  // Set dense matrix to zero
            }
        } else {
            // Dense column: place in dense matrix, rest set to zero in sparse matrix
            for (int row = 0; row < N; row++) {
                dense_output[row * d + col] = input[row * d + col];
                sparse_output[row * d + col] = __half(0.0f);  // Set sparse matrix to zero
            }
        }
    }
}

// Function to launch kernel
void split_matrix_columns(const half* input, half* dense_output, half* sparse_output,
                          int N, int d, int threshold) {  // threshold is an integer
    half *d_input, *d_dense_output, *d_sparse_output;
    
    // Allocate memory on GPU
    cudaMalloc(&d_input, N * d * sizeof(half));
    cudaMalloc(&d_dense_output, N * d * sizeof(half));
    cudaMalloc(&d_sparse_output, N * d * sizeof(half));

    // Copy data to GPU
    cudaMemcpy(d_input, input, N * d * sizeof(half), cudaMemcpyHostToDevice);

    // Launch kernel with grid size of d and block size of 32 (arbitrary)
    int blockSize = 32;
    int numBlocks = (d + blockSize - 1) / blockSize;
    sparse_column_split<<<numBlocks, blockSize>>>(d_input, d_dense_output, d_sparse_output, N, d, threshold);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy the results back to host
    cudaMemcpy(dense_output, d_dense_output, N * d * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(sparse_output, d_sparse_output, N * d * sizeof(half), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_dense_output);
    cudaFree(d_sparse_output);
}

int main() {
    int N = 5; // Number of rows
    int d = 4; // Number of columns
    int threshold = 2; // Threshold for determining sparsity (number of non-zero elements)

    // Example input matrix (N x d) in fp16
    half h_input[] = {
        __float2half(1.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f),
        __float2half(0.0f), __float2half(2.0f), __float2half(0.0f), __float2half(0.0f),
        __float2half(0.0f), __float2half(0.0f), __float2half(3.0f), __float2half(0.0f),
        __float2half(4.0f), __float2half(0.0f), __float2half(0.0f), __float2half(0.0f),
        __float2half(0.0f), __float2half(0.0f), __float2half(5.0f), __float2half(0.0f)
    };

    // Allocate space for the output matrices
    half* dense_output = new half[N * d];
    half* sparse_output = new half[N * d];

    // Call the kernel function
    split_matrix_columns(h_input, dense_output, sparse_output, N, d, threshold);

    // Print the result matrices
    std::cout << "Dense Output:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << __half2float(dense_output[i * d + j]) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Sparse Output:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            std::cout << __half2float(sparse_output[i * d + j]) << " ";
        }
        std::cout << std::endl;
    }

    // Free host memory
    delete[] dense_output;
    delete[] sparse_output;

    return 0;
}
