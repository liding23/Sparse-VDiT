/*
 * File Name: sparse_dense_split.cu
 * 
 * Description: This file contains the implementation to split a given input matrix 
 *              (of shape N x d) into sparse and dense matrices based on the number 
 *              of zero elements in each column. 
 *              - If the number of zero elements in a column is greater than a 
 *                specified threshold, that column is classified as sparse.
 *              - If the number of zero elements in a column is less than the 
 *                threshold, that column is classified as dense.
 *              The file records the positions of sparse columns and separates 
 *              the input matrix into sparse and dense matrices accordingly.
 *
 * Author: Li Ding
 * Created on: 2025-03-12
 * Version: 1.0
 *
 * Licensed under the MIT License. See the LICENSE file for details.
 */

#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel to identify sparse columns and count the number of zeros in each column
__global__ void identify_sparse_columns(const float* matrix, int* sparse_columns, int* zero_counts, int rows, int cols, int threshold) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;  // The column index being processed
    if (col < cols) {
        int zero_count = 0;
        
        // Count the number of zeros in the column
        for (int row = 0; row < rows; ++row) {
            if (matrix[row * cols + col] == 0) {
                zero_count++;
            }
        }

        // Store the zero count
        zero_counts[col] = zero_count;

        // Mark the column as sparse if zero count exceeds the threshold
        if (zero_count > threshold) {
            sparse_columns[col] = 1;  // Mark as sparse
        } else {
            sparse_columns[col] = 0;  // Mark as dense
        }
    }
}

// CUDA kernel to split the matrix into sparse and dense columns
__global__ void split_matrix_by_sparsity(const float* matrix, float* dense_matrix, float* sparse_matrix, 
                                          const int* sparse_columns, int* sparse_col_indices, int rows, int cols, int* sparse_col_count) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;  // The column index being processed
    if (col < cols) {
        if (sparse_columns[col] == 1) {  // Sparse column
            // Mark the column as sparse and save the non-zero elements in sparse_matrix
            int sparse_idx = atomicAdd(sparse_col_count, 1);  // Get the next available sparse index
            sparse_col_indices[sparse_idx] = col;  // Save column index
            for (int row = 0; row < rows; ++row) {
                float val = matrix[row * cols + col];
                if (val != 0) {
                    sparse_matrix[sparse_idx * rows + row] = val;  // Store non-zero value in sparse matrix
                }
            }
        } else {  // Dense column
            // Copy entire column to dense matrix
            for (int row = 0; row < rows; ++row) {
                dense_matrix[row * cols + col] = matrix[row * cols + col];
            }
        }
    }
}

// PyTorch interface function to identify sparse columns and split the matrix
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sparse_column_split_cuda(torch::Tensor matrix, int threshold) {
    int rows = matrix.size(0);
    int cols = matrix.size(1);

    // Allocate tensors for sparse columns, dense matrix, sparse matrix, and sparse column indices
    auto sparse_columns = torch::zeros({cols}, matrix.options().dtype(torch::kInt));
    auto zero_counts = torch::zeros({cols}, matrix.options().dtype(torch::kInt));
    auto dense_matrix = torch::zeros_like(matrix);
    auto sparse_matrix = torch::zeros({cols * rows}, matrix.options());  // Max possible sparse elements
    auto sparse_col_indices = torch::zeros({cols}, torch::dtype(torch::kInt));

    // Get raw pointers to the tensor data
    float* d_matrix = matrix.data_ptr<float>();
    int* d_sparse_columns = sparse_columns.data_ptr<int>();
    int* d_zero_counts = zero_counts.data_ptr<int>();
    float* d_dense_matrix = dense_matrix.data_ptr<float>();
    float* d_sparse_matrix = sparse_matrix.data_ptr<float>();
    int* d_sparse_col_indices = sparse_col_indices.data_ptr<int>();
    int* d_sparse_col_count = new int[1];  // Store the count of sparse columns

    cudaMemset(d_sparse_col_count, 0, sizeof(int));  // Initialize sparse column count to 0

    // Launch CUDA kernel to identify sparse columns
    int block_size = 256;
    int grid_size = (cols + block_size - 1) / block_size;
    identify_sparse_columns<<<grid_size, block_size>>>(d_matrix, d_sparse_columns, d_zero_counts, rows, cols, threshold);

    // Launch CUDA kernel to split matrix into sparse and dense columns
    split_matrix_by_sparsity<<<grid_size, block_size>>>(d_matrix, d_dense_matrix, d_sparse_matrix, 
                                                       d_sparse_columns, d_sparse_col_indices, rows, cols, d_sparse_col_count);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err)));
    }

    // Convert sparse matrix back to tensor with correct shape
    int sparse_count;
    cudaMemcpy(&sparse_count, d_sparse_col_count, sizeof(int), cudaMemcpyDeviceToHost);

    sparse_matrix = sparse_matrix.view({sparse_count, rows});

    return std::make_tuple(dense_matrix, sparse_matrix, sparse_col_indices);
}

// Bind the function to PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_column_split_cuda", &sparse_column_split_cuda, "Identify sparse columns and split the matrix into sparse and dense columns based on sparsity");
}
