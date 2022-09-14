#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>
#include <sys/time.h>

// Amxk, Bkxn, Cmxn
#define Scale 1024
#define Row_A Scale 
#define Col_A Scale
#define Col_B Scale

// 验证结果
bool is_equal(float *mat_a, float *mat_b, int num);
// 打印行优先矩阵
void print_mat_row(float *mat, int row, int col);
// 打印列优先矩阵
void print_mat_col(float* mat, int col, int row);
// 初始化矩阵
void mat_init (float *mat_a, float *mat_b, int m, int k, int n);
// CPU计算矩阵乘法
void cpu_mat_mul (float *mat_a, float *mat_b, float *mat_res, int m, int k, int n);
// GPU tensor core 加速计算矩阵乘法
void tensor_cublasSgemm(float *mat_a, float *mat_b, float *mat_res, int r_a, int c_a, int c_b, float alpha = 1.0, float beta = 0.0);


int main() {

/* 数据测试
    int m = 8, k = 16, n = 8;

    float A[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  };

    float B[] = {   1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    1, 2, 3, 4, 5, 6, 7, 8, 
                    };
    
    float *cpu_C = cpu_mat_mul(A, B, m, k, n);
    std::cout << "cpu_C:\n";
    print_mat_row(cpu_C, m, n); // 打印CPU计算结果
*/

    float *A = new float[Row_A * Col_A];
    float *B = new float[Col_A * Col_B];

    // 初始化矩阵 A B
    mat_init(A, B, Row_A, Col_A, Col_B);

    struct timeval cpu_start, cpu_end, gpu_start, gpu_end;
// cpu计算计时开始
    gettimeofday(&cpu_start, nullptr);
    float *cpu_C = new float[Row_A * Col_B];
    cpu_mat_mul(A, B, cpu_C, Row_A, Col_A, Col_B);
    gettimeofday(&cpu_end, nullptr);
    double cpu_time = (cpu_end.tv_sec*1e6 + cpu_end.tv_usec) - (cpu_start.tv_sec*1e6 + cpu_start.tv_usec); // us

/* 数据测试
    float *gpu_C = tensor_cublasSgemm(A, B, m, k, n, handle);
*/

// GPU计算计时开始
    gettimeofday(&gpu_start, nullptr);
    float *gpu_C = new float[Row_A * Col_B];
    tensor_cublasSgemm(A, B, gpu_C, Row_A, Col_A, Col_B);
    gettimeofday(&gpu_end, nullptr);
    double gpu_time = (gpu_end.tv_sec * 1e6 + gpu_end.tv_usec) - (gpu_start.tv_sec * 1e6 + gpu_start.tv_usec); // us

/* 数据测试
    std::cout << "gpu_C:\n";
    print_mat_col(gpu_C, n, m);
*/

/*
    // 验证结果
    float *gpu_C_T = new float[Row_A * Col_B];
    for (int i = 0; i < Col_B; i++) {
        for (int j = 0; j < Row_A; j++) {
            gpu_C_T[i * Row_A + j] = gpu_C[j * Col_B + i];
        }
    }
    std::cout << "compute result(cpu vs tensor): " << is_equal(cpu_C, gpu_C_T, Row_A * Col_B) << std::endl;
*/

    std::cout << "Matrix Scale: [" << Row_A << "][" << Col_A << "] * [" << Col_A << "][" << Col_B << "]\n"
    << "cpu_time: " << cpu_time / 1e6 << " s\n" << "gpu_time: " << gpu_time / 1e6 << " s\n" << "acc: " << cpu_time / gpu_time << std::endl;

    delete[] A;
    delete[] B;
    delete[] cpu_C;
    delete[] gpu_C;

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

// GPU tensor core 加速计算矩阵乘法
void tensor_cublasSgemm(float *mat_a, float *mat_b, float *mat_res, int r_a, int c_a, int c_b, float alpha, float beta) {
    /* 
        cublasSgemm(); 用于计算 C = α op(A)op(B) + β C；

        where α and β are scalars, and A, B and C are matrices stored in column-major format with dimensions op(A)mxk, op(B)kxn and Cmxn, respectively. Also, for matrix A:

        if (transa == 'CUBLAS_OP_N') {
            op(A) = A;
        } else if (transa == 'CUBLAS_OP_T') {
            op(A) = A_T;
        } else if (transa == 'CUBLAS_OP_C') {
            op(A) = A_H;
        }

        当α = 1, β = 0时，即为C(m,k) = A(m,n) * B(n,k)；
    */

    // 在显存中开辟空间
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, r_a * c_a * sizeof(float));
    cudaMalloc((void**)&d_B, c_a * c_b * sizeof(float));
    cudaMalloc((void**)&d_C, r_a * c_b * sizeof(float));

    // 将数据从CPU拷贝到GPU
    cublasSetVector(r_a * c_a, sizeof(float), mat_a, 1, d_A, 1);
    cublasSetVector(c_a * c_b, sizeof(float), mat_b, 1, d_B, 1);

    cublasHandle_t handle; // a pointer type to an opaque structure holding the cuBLAS library context.
    cublasCreate(&handle); // initializes the cuBLAS library and creates a handle to an opaque structure holding the cuBLAS library context.
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // 开启tensor core


    cublasSgemm(
        handle,
        CUBLAS_OP_T, // operation op(A) that is transpose
        CUBLAS_OP_T, // operation op(B) that is transpose
        r_a, // number of rows of matrix op(A) and C
        c_b, // number of columns of matrix op(B) and C
        c_a, // number of columns of op(A) and rows of op(B)
        &alpha, 
        d_A, 
        c_a, // leading dimension of two-dimensional array used to store the matrix A
        d_B, 
        c_b,  // leading dimension of two-dimensional array used to store matrix B
        &beta, 
        d_C, 
        r_a //leading dimension of a two-dimensional array used to store the matrix C
    );

    //将结果从GPU拷贝到CPU
    cublasGetVector(r_a * c_b, sizeof(float), d_C, 1, mat_res, 1);

    cublasDestroy(handle);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// 初始化矩阵
void mat_init (float *mat_a, float *mat_b, int m, int k, int n) {
    srand(0);
    // mat_a(m,k)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            mat_a[i * m + j] = random() % 10;
        }
    }
    // mat_b(k,n)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            mat_b[i * k + j] = random() % 10;
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

// 打印列优先矩阵
void print_mat_col(float* mat, int col, int row) {
    std::cout << std::endl;
    for (int i = 0; i < col; i++) {
        for (int j = 0; j < row; j++) {
            std::cout << mat[j * col + i] << " ";
        }
        std::cout << std::endl;
    }
}

// 验证结果
bool is_equal(float *mat_a, float *mat_b, int num) {
    for (int i = 0; i < num; i++) {
        if (mat_a[i] - mat_b[i] > (1.0e-10)) {
            return false;
        }
    }
    return true;
}
