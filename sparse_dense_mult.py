import torch
import sparse_column_split

# 创建一个示例矩阵
matrix = torch.tensor([[0, 1, 0, 3, 0],
                       [0, 0, 2, 5, 0],
                       [7, 0, 0, 9, 1],
                       [0, 0, 6, 0, 0]], dtype=torch.float32).cuda()


# 设置阈值
threshold = 2

# 调用自定义CUDA扩展
dense_matrix, sparse_matrix, sparse_col_indices = sparse_column_split.sparse_column_split_cuda(matrix, threshold)

# 输出结果
print("Dense matrix:")
print(dense_matrix)

print("Sparse matrix (non-zero values only):")
print(sparse_matrix)

print("Sparse column indices:")
print(sparse_col_indices)
