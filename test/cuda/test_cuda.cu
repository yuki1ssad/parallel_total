#include <random>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <sys/time.h>
#include <iostream>

// Amxk, Bkxn, Cmxn
#define Scale 1024
#define Row_A Scale 
#define Col_A Scale
#define Col_B Scale

#define Dimx_Num 32
#define Dimy_Num 32

// 验证结果
bool is_equal(float *mat_a, float *mat_b, int num);
// 打印行优先矩阵
void print_mat_row(float *mat, int row, int col);
// 初始化矩阵
void mat_init (float *mat_a, float *mat_b, int m, int k, int n);
// CPU计算矩阵乘法
void cpu_mat_mul(float *mat_a, float *mat_b, float *mat_res, int m, int k, int n);
// 定义核函数
__global__ void mat_mul_cuda_kernel(float *d_A, float *d_B, float *d_C, int m, int k, int n) ;
// GPU计算矩阵乘法
void gpu_mat_mul(float *mat_a, float *mat_b, float *mat_res, int m, int k, int n, int dimx_num, int dimy_num);
// 定义核函数（共享内存）
__global__ void mat_mul_cuda_kernel_shared(float *d_A, float *d_B, float *d_C, int m, int k, int n);
// GPU计算矩阵乘法（共享内存）
void gpu_mat_mul_shared(float *mat_a, float *mat_b, float *mat_res, int m, int k, int n, int dimx_num, int dimy_num);

int main() {
    float *A = new float[Row_A * Col_A];
    float *B = new float[Col_A * Col_B];

    // 初始化矩阵 A B
    mat_init(A, B, Row_A, Col_A, Col_B);

    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
    struct timeval gpu_start_s, gpu_end_s;

    // cpu_mat_mul
    gettimeofday(&cpu_start, nullptr);
    float *cpu_C = new float[Row_A * Col_B];
    cpu_mat_mul(A, B, cpu_C, Row_A, Col_A, Col_B);
    gettimeofday(&cpu_end, nullptr);
    double cpu_time = (cpu_end.tv_sec*1e6 + cpu_end.tv_usec) - (cpu_start.tv_sec*1e6 + cpu_start.tv_usec); //um

    // print_mat_row(cpu_C, Row_A, Col_B);

    // gpu_mat_mul
    gettimeofday(&gpu_start, nullptr);
    float *gpu_C = new float[Row_A * Col_B];
    gpu_mat_mul(A, B, gpu_C, Row_A, Col_A, Col_B, Dimx_Num, Dimy_Num);
    gettimeofday(&gpu_end, nullptr);
    double gpu_time = (gpu_end.tv_sec*1e6 + gpu_end.tv_usec) - (gpu_start.tv_sec*1e6 + gpu_start.tv_usec); //um

    // print_mat_row(gpu_C, Row_A, Col_B);

    // shared memory gpu_mat_mul
    gettimeofday(&gpu_start_s, nullptr);
    float *gpu_C_s = new float[Row_A * Col_B];
    gpu_mat_mul_shared(A, B, gpu_C_s, Row_A, Col_A, Col_B, Dimx_Num, Dimy_Num);
    gettimeofday(&gpu_end_s, nullptr);
    double gpu_time_shared = (gpu_end_s.tv_sec*1e6 + gpu_end_s.tv_usec) - (gpu_start_s.tv_sec*1e6 + gpu_start_s.tv_usec); //um

    // print_mat_row(gpu_C_s, Row_A, Col_B);

    // 验证结果
    // std::cout << "compute result(cpu vs gpu): " << is_equal(cpu_C, gpu_C, Row_A * Col_B) << "\ncompute result(gpu vs gpu_s): " << is_equal(cpu_C, gpu_C_s, Row_A * Col_B) << std::endl;


    // 输出结果
    std::cout << "矩阵规模：[" << Row_A  <<"][" << Col_A << "] * [" << Col_A << "][" << Col_B << "]\n";
    std::cout << "cpu_time = " << cpu_time/1e6 << " s\n" << "gpu_time = " << gpu_time/1e6 << " s\n" << "acc = " << cpu_time/gpu_time << std::endl;
    std::cout << "gpu_time_shared = " << gpu_time_shared/1e6 << " s\n" << "acc_shared_memory = " << cpu_time/gpu_time_shared << std::endl;

    delete[] cpu_C;
    delete[] gpu_C;
    delete[] gpu_C_s;

    return 0;
}


// CPU计算矩阵乘法
void cpu_mat_mul (float *mat_a, float *mat_b, float *mat_res, int m, int k, int n) { // mat_a(m,k) * mat_b(k,n)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int tmp = 0;
            for (int q = 0; q < k; q++) {
                tmp += mat_a[i * k + q] * mat_b[q * n + j];
                // std::cout << tmp << " " << mat_a[i * k + q] << " " << mat_b[q * n + j] << std::endl;
            }
            mat_res[i * n + j] = tmp;
        }
    }
}

// 初始化矩阵
void mat_init (float *mat_a, float *mat_b, int m, int k, int n) {
    srand(0);
    // mat_a(m,k)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            mat_a[i * m + j] = float(random() % 10);
        }
    }
    // mat_b(k,n)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            mat_b[i * k + j] = float(random() % 10);
        }
    }
}

// 打印行优先矩阵
void print_mat_row(float *mat, int row, int col) {
    std::cout << std::endl;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            std::cout << mat[i * col + j] <<" ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 核函数
__global__ void mat_mul_cuda_kernel(float *d_A, float *d_B, float *d_C, int m, int k, int n) {
    int ix = threadIdx.x + blockDim.x*blockIdx.x; // col
    int iy = threadIdx.y + blockDim.x*blockIdx.y; // row
    
    if (ix < n && iy < m) {
        float tmp = 0;
        for (int q = 0; q < k; q++) {
            tmp += d_A[iy * k + q] * d_B[q * n + ix];
        }
        d_C[iy * n + ix] = tmp;
    }
}

// GPU计算矩阵乘法
void gpu_mat_mul(float *mat_a, float *mat_b, float *mat_res, int m, int k, int n, int dimx_num, int dimy_num) {
    // 从CPU拷贝数据到GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, mat_a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, mat_b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // 设置参数
    dim3 block(dimx_num, dimy_num);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    mat_mul_cuda_kernel<<<grid, block>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();
    cudaMemcpy(mat_res, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 核函数（共享内存）
__global__ void mat_mul_cuda_kernel_shared(float *d_A, float *d_B, float *d_C, int m, int k, int n) {
    __shared__ float shared_A[Dimy_Num][Dimx_Num];
    __shared__ float shared_B[Dimx_Num][Dimy_Num];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * Dimy_Num + ty;
    int col = bx * Dimx_Num + tx;

    float tmp = 0.;
    for (int i = 0; i < (k+Dimx_Num - 1) / Dimx_Num; i++) {
        if (i * Dimx_Num + tx < k && row < m) {
            shared_A[ty][tx] = d_A[row * k + i * Dimx_Num + tx];
        } else {
            shared_A[ty][tx] = 0.;
        }

        if (i * Dimy_Num + ty < k && col <n) {
            shared_B[ty][tx] = d_B[(i * Dimy_Num + ty) * n + col];
        } else {
            shared_B[ty][tx] = 0.;
        }
        __syncthreads();

        for (int j = 0; j < Dimx_Num; j++) {
            tmp += shared_A[ty][j] * shared_B[j][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        d_C[row * n + col] = tmp;
    }
}

// GPU计算矩阵乘法（共享内存）
void gpu_mat_mul_shared(float *mat_a, float *mat_b, float *mat_res, int m, int k, int n, int dimx_num, int dimy_num) {
    // 从CPU拷贝数据到GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, mat_a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, mat_b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // 设置参数
    dim3 block(dimx_num, dimy_num);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    mat_mul_cuda_kernel_shared<<<grid, block>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();
    cudaMemcpy(mat_res, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 验证结果
bool is_equal(float *mat_a, float *mat_b, int num) {
    for (int i = 0; i < num; i++) {
        if (fabs(mat_a[i] - mat_b[i]) > (1.0e-10)) {
            return false;
        }
    }
    return true;
}
