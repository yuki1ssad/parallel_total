#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

#define Mat_Scale 2048
#define Epoch 10

double A[Mat_Scale][Mat_Scale], B[Mat_Scale][Mat_Scale], C[Mat_Scale][Mat_Scale];



//初始化矩阵
void init_mat(int a_r, int a_c, int b_r, int b_c) {
    srand((unsigned)time(NULL));

    for (int i = 0; i < a_r; i++) {
        for (int j = 0; j < a_c; j++) {
            A[i][j] = rand() % 10;
        }
    }

    for (int i = 0; i < b_r; i++) {
        for (int j = 0; j < b_c; j++) {
            B[i][j] = rand() % 10;
        }
    }
    
    return;
}

void mat_mul_parallel(int a_r, int a_c,int b_c, int thread) {
    omp_set_num_threads(thread);
    #pragma omp parallel for
    
    for (int i = 0; i < a_r; i++) {
        for (int j = 0; j < b_c; j++) {
            double sum = 0.0;
            for (int k = 0; k < a_c; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return;
}

void mat_mul(int a_r, int a_c, int b_c) {
    for (int i = 0; i < a_r; i++) {
        for (int j = 0; j < b_c; j++) {
            double sum = 0.0;
            for (int k = 0; k < a_c; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return;
}

//计算量


//矩阵规模
void Exp_Scale(int epoch, ofstream &outfile) {
    struct timeval p_s_t, p_e_t, s_t, e_t;//并行开始、结束时间，非并行开始、结束时间
    for (int s = 64; s <= 2048; s *= 2) { //不同矩阵规模
        init_mat(s, s, s, s);
        cout << "Scale: " << s << " * " << s << endl;
        for (int trd = 2; trd <= 24; trd += 2) {
            //并行
            gettimeofday(&p_s_t,NULL);

            mat_mul_parallel(s,s,s,trd);

            gettimeofday(&p_e_t, NULL);
            double parallel_t = (p_e_t.tv_sec*1000000 + p_e_t.tv_usec) - (p_s_t.tv_sec*1000000 + p_s_t.tv_usec);
        
            //非并行
            gettimeofday(&s_t,NULL);
            mat_mul(s,s,s);
            gettimeofday(&e_t,NULL);
            double t = (e_t.tv_sec*1000000 + e_t.tv_usec) - (s_t.tv_sec*1000000 + s_t.tv_usec);

            cout << "线程数 = " << trd << endl << "parallel_t = " << parallel_t/1000000 << "s" << endl
                << "         t = " << t/1000000 << "s" << endl << "加速比为：" << t/parallel_t << endl << endl;

            outfile << s << "," << trd << "," << t/parallel_t << endl;
        }
    }

}

int main() {

    ofstream outfile;
    outfile.open("data_test1.csv");
    outfile << "SCALE,THREAD,RATIO" << endl;

    int epoch = Epoch;
    cout << "Epoch:" << epoch << endl;
    Exp_Scale(epoch, outfile);

    outfile.close();

    return 0;
}