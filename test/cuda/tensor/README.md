# Martix_Multiplication with Tensor Core by cuBLAS
The examples are written for Linux, not Windows.

# How-to

## For demo1_tensorcore_acc.cu
### Compilation
The simplest way to configure is to run command:
```
nvcc -o demo1 demo1_tensorcore_acc.cu 
```

### Run
The following command runs the compiled codes.

```
./demo1

```
### Result
<img src="Cpp_test/parallel_computing/test/cuda/tensor_core_acc.png" alt="tensor_core_acc.png">

