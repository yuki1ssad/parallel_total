#include<iostream>
#include<mpi.h>
#include<math.h>
#include"omp.h"
#include<stdlib.h>
#include<time.h>
// #include<Windows.h>
#include <unistd.h> 
#include <fstream>

using namespace std;  // proc 8 12 16
#define MSIZE 512  // 512 1024 2048
#define NUM_THREADS 1  // 8 12 16
void initMatrix(int* A, int rows, int cols);
void matMultiply(int* A, int* B, int* result, int m, int p, int n, int num, int rank);

void print_results(const char* prompt, int a[MSIZE][MSIZE])
{
	int i, j;

	printf("\n\n%s\n", prompt);
	for (i = 0; i < MSIZE; i++) {
		for (j = 0; j < MSIZE; j++) {
			printf(" %d", a[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
}
int main()
{
	double seq_time, par_time;
	int m = MSIZE, n = MSIZE, p = MSIZE;
	double start, stop;
	int* A = NULL, *B = NULL, *C = NULL;
	int* bA, *bC;
	int rank, num;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int bm = m / num;

	bA = new int[bm * p];
	B = new int[p * n];
	bC = new int[bm * n];
	A = new int[m * p];
	C = new int[m * n];

	//��ʼ������
	if (rank == 0) {
		initMatrix(A, m, p);
		unsigned sleep(1000);  //���Ͼ���A,B��Ԫ�ز���ȫ��ͬ
		initMatrix(B, p, n);

// 		// int seq_A[MSIZE][MSIZE] = { 0 };
// 		int* seq_A = new int[MSIZE * MSIZE];
// 		// int	seq_B[MSIZE][MSIZE] = { 0 };
// 		int* seq_B = new int[MSIZE * MSIZE];
// 		// int seq_C[MSIZE][MSIZE];
// 		int* seq_C = new int[MSIZE * MSIZE];

// 		int i = 0;
// 		int j = 0;
// 		for (int m = 0; m < MSIZE * MSIZE; m++)
// 		{
// 			i = m / MSIZE;
// 			j = m % MSIZE;
// 			seq_A[i][j]= A[m];
// 			seq_B[i][j] = B[m];

// 		}

// 		double start1, end1;  //seq��ʼ�ͽ���ʱ��
// 		start1 = MPI_Wtime();

// // #pragma omp parallel for num_threads(NUM_THREADS)
// 		for (int i = 0; i < MSIZE; i++) {
// 			for (int j = 0; j < MSIZE; j++) {
// 				seq_C[i][j] = 0;
// 				for (int k = 0; k < MSIZE; k++) {
// 					seq_C[i][j] += seq_A[i][k] * seq_B[k][j];
// 				}
// 			}
// 		}
// 		end1 = MPI_Wtime();

		// seq_time = end1 - start1;


		//for (int i = 0; i < MSIZE * MSIZE; i++) {

		//	cout << A[i] << " ";
		//	if (i % MSIZE == MSIZE - 1)
		//		cout << endl;

		//}
		//cout << endl;

		//for (int i = 0; i < MSIZE * MSIZE; i++) {

		//	cout << B[i] << " ";
		//	if (i % MSIZE == MSIZE - 1)
		//		cout << endl;

		//}
		//cout << endl;

	}

	MPI_Barrier(MPI_COMM_WORLD);
	start = MPI_Wtime();
	//�ַ�����
	MPI_Scatter(A, bm * p, MPI_FLOAT, bA, bm * p, MPI_FLOAT, 0, MPI_COMM_WORLD);
	MPI_Bcast(B, p * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//����˷�
	matMultiply(bA, B, bC, bm, p, n, num, rank);
	MPI_Barrier(MPI_COMM_WORLD);
	//�������
	MPI_Gather(bC, bm * n, MPI_FLOAT, C, bm * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//ʣ��ֿ����
	int remainRowsStartId = bm * num;               // 10*10   4   10%4=2   2*4=8 < 10   N%m = 0
	if (rank == 0 && remainRowsStartId < m) {
		int remainRows = m - remainRowsStartId;
		matMultiply(A + remainRowsStartId * p, B, C + remainRowsStartId * n,
			remainRows, p, n, num, rank);


	}
	//if (rank == 0) {
	//	for (int i = 0; i < MSIZE * MSIZE; i++) {

	//		cout << C[i] << " ";
	//		if (i % MSIZE == MSIZE - 1)
	//			cout << endl;

	//	}
	//	cout << endl;
	//}

	stop = MPI_Wtime();

	fstream outfile;
    outfile.open("data_mpi2_yl.csv",ios::out | ios::app);

	if (rank == 0) {
		par_time  = stop - start;
		/*cout << "MPI����ʱ��Ϊ��" << par_time << endl;*/
		// double acc = seq_time / par_time;
		cout << "par_time = " << par_time << endl;
		outfile << endl << num << "," << NUM_THREADS << "," << par_time;
	}

	MPI_Finalize();

	delete[] A;
	delete[] B;
	delete[] C;
	delete[] bA;
	delete[] bC;

	outfile.close();

	return 0;
}

void initMatrix(int* A, int rows, int cols)
{
	srand((unsigned)time(NULL));

	for (int i = 0; i < rows * cols; i++) {
		A[i] = rand() % 10 + 1;
	}
}

void matMultiply(int* A, int* B, int* result, int m, int p, int n, int num, int rank)
{
#pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < m; i++) {

		for (int j = 0; j < n; j++) {
			int temp = 0;
			for (int k = 0; k < p; k++) {

				temp += A[i * p + k] * B[k * n + j];

				/*cout << "i=��" << i << " ";
				cout << "m=��" << m << " ";
				cout << "j=��" << j << " ";
				cout << "n=��" << n << " ";
				cout << "k=��" << k << " ";
				cout << "p=��" << p << " " <<endl;*/

				//for (int tag = 0; tag < num; tag++) {
				//	if (tag == rank) {
				//		cout << "�����ǽ��̣�" << tag << "�ڷ��� ";
				//		cout << "B�ĵ�" << k * n + j << "��Ԫ��" << B[k * n + j] << " ";

				//		cout << endl;

				//		int* oresult = result;
				//		/*cout << "ԭ���Ļ����ǣ�" << result << endl;*/
				//		if(j == 2 || j==3){
				//			
				//			cout << "ԭ���Ļ����ǣ�" << result << endl;

				//			cout << endl;
				//			cout << "�����ǲ��Ǳ��ˣ�" << result << endl;

				//		}
				//	}
				//	else
				//	{
				//		continue;
				//	}
				//}
						/*cout << "����B�ĵ�" << k * n + j << "��Ԫ��" << B[k * n + j] << " ";*/
			}
		

			result[i * n + j] = temp;
			/*cout << endl;
			cout << "���ڻ���ĵ�ַ��" << result << "��װ�ŵ���" << result[i * n + j];
			cout << endl;*/
		}
		

		//if (i == m - 1)
		//	cout << endl;
		//cout << "������һ�����̵Ļ�����" << endl;
	}

	//mpiexec -n 3 Project2.exe
}