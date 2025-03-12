from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='sparse_column_split',
    ext_modules=[cpp_extension.CUDAExtension(
        'sparse_column_split',
        ['sparse_column_split.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
