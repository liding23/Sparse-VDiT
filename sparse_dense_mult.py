import torch
from torch.utils.cpp_extension import load

# Load and compile the CUDA extension
sparse_dense = load(
    name="sparse_dense",
    sources=["sparse_dense_matrix_multiply.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"]
)

class SparseDenseMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, num_sparse_cols):
        num_rows, num_cols = A.shape
        num_dense_cols = num_cols - num_sparse_cols
        C = torch.zeros((num_rows, num_cols), device=A.device, dtype=A.dtype)

        # Call the CUDA kernel
        sparse_dense.matrix_multiply(A, B, C, num_rows, num_cols, num_sparse_cols, num_dense_cols)

        # Save context for backward pass
        ctx.save_for_backward(A, B)
        ctx.num_sparse_cols = num_sparse_cols

        return C

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        num_sparse_cols = ctx.num_sparse_cols
        # Compute gradients (left as an exercise if required)
        grad_A = grad_B = None
        return grad_A, grad_B, None

# Wrapper function to integrate into PyTorch
def sparse_dense_matmul(A, B, num_sparse_cols):
    return SparseDenseMatMul.apply(A, B, num_sparse_cols)
