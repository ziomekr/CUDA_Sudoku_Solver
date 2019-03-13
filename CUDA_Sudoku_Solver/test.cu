#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sudoku_GPU.cuh"
#include "sudoku_CPU.h"
#include <ctime>
#include <iostream>


void print(char* sudoku) {
	printf("-----------------------------------------------------------------------\n");
	for (int i = 0; i < DIMENSION*DIMENSION; i++) {
		printf("%d ", sudoku[i]);
		if (i%DIMENSION == DIMENSION - 1)
			printf("\n");
	}
	printf("-----------------------------------------------------------------------\n");
}
void solve_sudoku(char* sudoku) {
	print(sudoku);
	print(solve_sudokuGPU(sudoku));
	std::clock_t c_start = std::clock();
	solve_sudokuCPU(sudoku);
	std::clock_t c_end = std::clock();
	long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	std::cout << "CPU time used: " << time_elapsed_ms << " ms\n";
	print(sudoku);
};
int main()
{
	char sudoku[81] = { 5,3,0,0,7,0,0,0,0,6,0,0,1,9,5,0,0,0,0,9,8,0,0,0,0,6,0,8,0,0,0,6,0,0,0,3,4,0,0,8,0,3,0,0,1,7,0,0,0,2,0,0,0,6,0,6,0,0,0,0,2,8,0,0,0,0,4,1,9,0,0,5,0,0,0,0,8,0,0,7,9 };
	char sudoku1[81] = { 0,0,3,0,0,4,0,0,0,5,1,0,0,0,0,0,0,0,0,8,0,3,2,0,0,0,4,0,0,2,0,1,7,0,0,0,1,0,0,0,0,0,0,0,7,0,0,0,5,8,0,4,0,0,6,0,0,0,3,2,0,9,0,0,0,0,0,0,0,0,6,2,0,0,0,9,0,0,5,0,0 };
	char sudoku2[81] = { 8,0,0,0,0,0,0,0,0,0,0,3,6,0,0,0,0,0,0,7,0,0,9,0,2,0,0,0,5,0,0,0,7,0,0,0,0,0,0,0,4,5,7,0,0,0,0,0,1,0,0,0,3,0,0,0,1,0,0,0,0,6,8,0,0,8,5,0,0,0,1,0,0,9,0,0,0,0,4,0,0 };
	char sudoku3[81] = { 9,0,0,0,4,0,0,0,3,0,0,1,0,0,0,2,0,0,0,3,0,7,0,2,0,4,0,0,0,9,0,0,0,1,0,0,1,0,0,0,3,0,0,0,9,0,0,8,0,0,0,7,0,0,0,1,0,5,0,8,0,6,0,0,0,6,0,0,0,5,0,0,2,0,0,0,6,0,0,0,7 };
	char sudoku4[81] = {1,0,0,0,0,7,0,9,0,0,3,0,0,2,0,0,0,8,0,0,9,6,0,0,5,0,0,0,0,5,3,0,0,9,0,0,0,1,0,0,8,0,0,0,2,6,0,0,0,0,4,0,0,0,3,0,0,0,0,0,0,1,0,0,4,0,0,0,0,0,0,7,0,0,7,0,0,0,3,0, 0};
	
	solve_sudoku(sudoku);
	solve_sudoku(sudoku1);
	solve_sudoku(sudoku2);
	solve_sudoku(sudoku3);
	solve_sudoku(sudoku4);
	
	return 0;

}