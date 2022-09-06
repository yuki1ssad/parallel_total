#include <stdio.h>
#include<iostream>
#include<time.h>
#include<mpi.h>
#include<omp.h>
using namespace std;

#define matrix_scale 2048  //矩阵规模 256 512 1024 2048
#define threads 4        //线程数2 4 8 16 32 64

//结果显示
void show(double* matrix)
{
    for (int i = 0; i < matrix_scale * matrix_scale; i++)
    {
        if (i % (matrix_scale) == 0 && i != 0)
            printf("\n");
        printf("%f ", matrix[i]);
    }
    printf("\n");
}
//矩阵随机初始化
void Init(double* matrix)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < matrix_scale * matrix_scale; i++)
        matrix[i] = rand() / (double)(RAND_MAX / 100);
}
//定义数据类型
struct Message {
    int row_begin;              //起始行号，计算的行为[row_begin,row_end]
    int row_end;                //截止行号
    double res[matrix_scale*threads]; //需要暂存threads行计算结果
};

int main(int argc, char* argv[])
{
    clock_t start, end; //计时 起始，终止
    double time;        //所用时间
    int rank, size;     //进程号，通信域大小   
    int row_number;     // 行号
    MPI_Status stat;    //MPI状态，主要用来记录当前哪个进程结束计算，
    Message result;     //传递结果，包括起始行，终止行，2行计算结果信息

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype Messagetype;//声明Messagetype类型
    MPI_Datatype oldtype[2] = { MPI_INT,MPI_DOUBLE };   //在Messagetype中有两种数据类型，int与double
    int blockcount[2] = { 2,matrix_scale*threads };     //Messagetype中两种数据类型各有多少个
    MPI_Aint offset[2] = { 0,2 * sizeof(int) };         //两种数据类型的偏移为多少
    MPI_Type_create_struct(2, blockcount, offset, oldtype, &Messagetype);   
    MPI_Type_commit(&Messagetype);

    double* matrix_a = (double*)malloc(matrix_scale * matrix_scale * sizeof(double));
    double* matrix_b = (double*)malloc(matrix_scale * matrix_scale * sizeof(double));
    double* matrix_c = (double*)malloc(matrix_scale * matrix_scale * sizeof(double));

    Init(matrix_a);
    Init(matrix_b);

    if (rank == 0)                  //0号进程为主进程，主要进程任务的分发与接收
    {
        //show(matrix_a);
        int count = 0;              //在计算的进程数  
        int row = 0;                //需要处理第row行

        start = clock();            //开始计时

        for (int process = 1; process < size; process++)    //先给除0号进程之外的线程分发任务，每个进程拿到threads行去计算
        {
            MPI_Send(&row, 1, MPI_INT, process, 0, MPI_COMM_WORLD);
            count++;                                        //可用计算进程数+1
            row+=threads;
        }

        //printf("count====%d\n", count);
        do {                                                    //当可用计算进程数为0时结束计算
            MPI_Recv(&result, 1, Messagetype, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &stat);
                                                               //收回结果，结果保存在result中，类型为Messagetype
            --count;                                          //当前在计算的进程数减一

            int num = 0;
            for (int i = result.row_begin; i <= result.row_end; i++)    //向矩阵c写入数据
                for (int j = 0; j < matrix_scale; j++)                  //每一行的每一列
                    matrix_c[i * matrix_scale + j] = result.res[num++]; //res数据是从0开始

            if (row < matrix_scale)                                     //如果当前仍有剩余行
            {
                MPI_Send(&row, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);     
                                                                        //分配新任务给刚计算完的进程，进程会话tag为1，表示仍需计算
                count++;                                                //计算进程数+1
                row+=threads;                                           //下次分配的行号为row+threaads
            }
            else                                                        //所有数据已分配分完
                MPI_Send(&row, 1, MPI_INT, stat.MPI_SOURCE, 2, MPI_COMM_WORLD);    
                                                                        //进程会话tag为2，表示结束计算
        } while (count > 0);

        /*printf("show C\n");                                           //查看结果正确与否
        show(matrix_c);*/
        
        end = clock();                                                   //结束计时
        time = (end - start) * 1.0 / CLOCKS_PER_SEC;
        printf("矩阵规模：%d*%d，进程数：%d，线程数：%d,MPI+OpenMP执行时间为：%f\n", matrix_scale, matrix_scale, size, threads,time);
        free(matrix_a);
        free(matrix_b);
        free(matrix_c);
    }
    else
    {
        int r_num;                                                              //记录当前需要计算的起始行号
        MPI_Recv(&r_num, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);    //接收起始行号，会话tag可能为0,或2
        while (stat.MPI_TAG == 0)                                               //如果进程会话的tag为0，则表示有新任务分配
        {
            result.row_begin = r_num;                                           //记录起始行以便0号进程将计算结果写入matrix_c
            result.row_end = r_num + threads-1;                                 //记录终止行
            if (r_num + 1 >= matrix_scale)                                      //如果终止行超过了矩阵范围，则将其设置为最后一行
                result.row_end = matrix_scale - 1;
            int number = 0;                                                     //暂存的计算结果下标从0开始，
#pragma omp parallel for num_threads(threads)
            for (int row = result.row_begin; row <= result.row_end; row++)      //计算区间[begin,end]
            {
                for (int col = 0; col < matrix_scale; col++)
                {
                    result.res[number]=0;
                    for (int k = 0; k < matrix_scale; k++)
                    {
                        result.res[number] += matrix_a[row * matrix_scale + k] * matrix_b[k * matrix_scale + col];
                    }
                    number++;                                                   
                }
            }

            MPI_Send(&result, 1, Messagetype, 0, 1, MPI_COMM_WORLD);        //返回result，包含起始行，终止行以及当前计算结果
            MPI_Recv(&r_num, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);//等待接收新任务，进程会话tag可能为0或2
        }

    }
    MPI_Finalize();                             //清除MPI环境

    return 0;
}