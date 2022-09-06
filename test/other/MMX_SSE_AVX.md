```c++
#include <x86intrin.h>
//#include <intrin.h>    //(include immintrin.h)
#include <ctime>
#include <iostream>

#define ElementType int
#define N 2048

using namespace std;
typedef struct {
    int height;
    int width;
    int stride;
    ElementType * data;  //二维矩阵用一维数组保存
}Matrix;


// 设置矩阵中的元素
void setElement(Matrix A, int row, int column, ElementType val){
    A.data[row * A.width + column] = val;
}

// 矩阵元素随机初始化
void randomMatrix(Matrix A){
    for (int i = 0; i < A.width; i++)
        for (int j = 0; j < A.height; j++)
            setElement(A, i, j, rand() % 10 * 1.1);
}



void mat_add(Matrix m1, Matrix m2, Matrix m3){
    for (int i = 0; i < m3.height; ++i) {
        for (int j = 0; j < m3.width; ++j) {
            m3.data[i * m3.width + j] = m1.data[i * m1.width + j] + m2.data[i * m2.width + j];
        }
    }
}

void mat_add_MMX(Matrix m1, Matrix m2, Matrix m3){
    const int32_t* q;
    for (int i = 0; i < m3.height; ++i) {
        for (int j = 0; j < m3.width - 1; j += 2) {
            // load
            __m64 p1 = _mm_set_pi32(m1.data[i * m1.width + j], m1.data[i * m1.width + j + 1]);
            __m64 p2 = _mm_set_pi32(m2.data[i * m2.width + j], m2.data[i * m2.width + j + 1]);
            // compute
            __m64 res = _m_paddd(p1, p2);
            // store
            q = (const int32_t *)&res;
            m3.data[i * m3.width + j] = q[0];
            m3.data[i * m3.width + j + 1] = q[1];
        }
    }
}

void mat_add_SSE(Matrix m1, Matrix m2, Matrix m3){
    for (int i = 0; i < m3.height; ++i) {
        for (int j = 0; j < m3.width; j += 4) {
            __m128 p1 = _mm_load_ps(reinterpret_cast<const float *>(&m1.data[i * m1.width + j]));   //load
            __m128 p2 = _mm_load_ps(reinterpret_cast<const float *>(&m2.data[i * m2.width + j]));
            __m128 res = _mm_add_ps(p1, p2);   //compute
            _mm_store_ps(reinterpret_cast<float *>(&m3.data[i * m3.width + j]), res);  //store
        }
    }
}

void mat_add_AVX256(Matrix m1, Matrix m2, Matrix m3){
    for (int i = 0; i < m3.height; ++i) {
        for (int j = 0; j < m3.width; j += 8) {
            __m256 p1 = _mm256_load_ps(reinterpret_cast<const float *>(&m1.data[i * m1.width + j]));  //load
            __m256 p2 = _mm256_load_ps(reinterpret_cast<const float *>(&m2.data[i * m2.width + j]));
            __m256 res = _mm256_add_ps(p1, p2);  //compute
            _mm256_store_ps(reinterpret_cast<float *>(&m3.data[i * m3.width + j]), res);  //store
        }
    }
}


void print_matrix(Matrix m){
    for (int i = 0; i < m.height; ++i) {
        for (int j = 0; j < m.width; ++j) {
            cout<<m.data[i * m.width + j]<<"  "<<flush;
        }
        cout<<"\n"<<flush;
    }
}

int main(int argv, char *args[])
{
    clock_t start, end;

    // 初始化矩阵A、B、C
    Matrix A, B, C, C2;
    A.width = A.stride = N; A.height = N;
    A.data = new ElementType[N*N];
    randomMatrix(A);
    B.width = B.stride = N; B.height = N;
    B.data = new ElementType[N*N];
    randomMatrix(B);
    C.width = C.stride = N; C.height = N;
    C.data = new ElementType[N*N];
    C2.width = C2.stride = N; C2.height = N;
    C2.data = new ElementType[N*N];

    // 矩阵加法
    start = clock();
    for (int i = 0;i < 100; i++)
      mat_add(A, B, C);
    end = clock();
    cout<< "mat add is : " << (double)(end - start) / CLOCKS_PER_SEC<<endl;

    // MMX
    start = clock();
    for (int i = 0;i < 100; i++)
      mat_add_MMX(A, B, C2);
    end = clock();
    cout<< "MMX mat add is : " << (double)(end - start) / CLOCKS_PER_SEC<<endl;

    // SSE
    start = clock();
    for (int i = 0;i < 100; i++)
      mat_add_SSE(A, B, C2);
    end = clock();
    cout<< "SSE mat add is : " << (double)(end - start) / CLOCKS_PER_SEC<<endl;

    // AVX
    start = clock();
    for (int i = 0;i < 100; i++)
      mat_add_AVX256(A, B, C);
    end = clock();
    cout<< "AVX mat add is : " << (double)(end - start) / CLOCKS_PER_SEC<<endl;

    return 0;
}
```

