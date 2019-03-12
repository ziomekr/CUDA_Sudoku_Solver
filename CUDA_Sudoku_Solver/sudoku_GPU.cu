#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "sudoku_GPU.cuh"


#define DIMENSION 9
#define SUBSQUARE_DIMENSION 3
#define BLOCK_DIMENSION 256
#define GRID_DIMENSION 1024*5
#define MAX_SUDOKUS_COUNT 5000000


__device__
int row_used_numbers(char* sudokus_arr, int row, int* masks) {
	int used_numbers = 0;
	for (int i = row; i < row + DIMENSION; i++) {
		used_numbers = used_numbers | masks[sudokus_arr[i]];
	}
	return used_numbers;
}

__device__
int column_used_numbers(char* sudokus_arr, int column, int* masks) {
	int used_numbers = 0;
	for (int i = 0; i < DIMENSION; i++) {
		used_numbers = used_numbers | masks[sudokus_arr[column + i * DIMENSION]];
	}
	return used_numbers;
}

__device__
int subsquare_used_numbers(char* sudokus_arr, int subsquare_top_left_cell, int* masks) {
	int used_numbers = 0;
	for (int i = subsquare_top_left_cell; i < subsquare_top_left_cell + SUBSQUARE_DIMENSION; i++) {
		for (int j = 0; j < SUBSQUARE_DIMENSION; j++) {
			used_numbers = used_numbers | masks[sudokus_arr[i + j * DIMENSION]];
		}
	}
	return used_numbers;
}

__device__
int get_valid_numbers(char* sudokus_arr, int cell, int tIdx, int* masks) {
	int board_offset = cell % DIMENSION;
	int top_left_sudoku_cell = tIdx * DIMENSION*DIMENSION;
	int row = cell - board_offset;
	int column = top_left_sudoku_cell + board_offset;
	int subsquare_top_left_cell = cell - (cell % SUBSQUARE_DIMENSION) - (((cell - top_left_sudoku_cell) / DIMENSION) % SUBSQUARE_DIMENSION)*DIMENSION;
	return (row_used_numbers(sudokus_arr, row, masks) | column_used_numbers(sudokus_arr, column, masks) | subsquare_used_numbers(sudokus_arr, subsquare_top_left_cell, masks));
}


__device__
void copy_sudoku(char* src_sudoku, char* dest_sudoku) {
	for (int i = 0; i < DIMENSION*DIMENSION; i++) {
		*(dest_sudoku + i) = *(src_sudoku + i);
	}
}
__device__ void fill_masks(int* masks) {
	*masks = 0;
	for (int i = 0; i < DIMENSION; i++) {
		*(masks + i + 1) = 2 << i;
	}
}

__global__
void generate_next_permutations(char* sudokus_arr, char* sudoku_arr_new_permutations, int* number_of_old_permutations, int* number_of_permutations, int empty_cell){

	int tIdx = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int masks[DIMENSION + 1];
	__shared__ int valid_numbers[BLOCK_DIMENSION];
	
	fill_masks(masks);


	while (tIdx < *number_of_old_permutations) {
		
		
		valid_numbers[threadIdx.x] = get_valid_numbers(sudokus_arr, tIdx * DIMENSION * DIMENSION + empty_cell, tIdx, masks);
	
		
		for (int i = 1; i < DIMENSION + 1; i++) {
			if ((masks[i] & (~valid_numbers[threadIdx.x])) != 0) {
				int id = atomicAdd(number_of_permutations, 1);
				copy_sudoku(sudokus_arr + tIdx * DIMENSION*DIMENSION, sudoku_arr_new_permutations + id * DIMENSION*DIMENSION);
				sudoku_arr_new_permutations[id*DIMENSION*DIMENSION + empty_cell] = (char)i;
			}
		}
		tIdx += blockDim.x * gridDim.x;

	}
	
	
}

__global__ void backtrackigKernel(char * sudokus_arr, int number_of_permutations, int current_sudoku_index, int empty_cell)
{
	return void();
}


void err() {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));

	}
}

int get_empty_indices(char* sudoku, int* empty) {
	int count = 0;
	for (int i = 0; i < DIMENSION*DIMENSION; i++) {
		if ((int)sudoku[i] == 0) {
			empty[count++] = i;
		}
	}
	return count;
}
char* solve_sudokuGPU(char* sudoku) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	int count;
	int empty[DIMENSION*DIMENSION];
	int *old_perm, *new_perm;
	char *n, *p;
	char *result;
	result = (char*)malloc(sizeof(char) * 81);
	int np = 1;
	cudaEventRecord(start);
	cudaMalloc((void**)&n, MAX_SUDOKUS_COUNT * DIMENSION * DIMENSION * sizeof(char));
	err();
	cudaMalloc((void**)&p, MAX_SUDOKUS_COUNT * DIMENSION * DIMENSION * sizeof(char));
	err();

	cudaMalloc((void**)&old_perm, sizeof(int));
	
	cudaMalloc((void**)&new_perm, sizeof(int));
	cudaMemcpy(n, sudoku, DIMENSION * DIMENSION * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(new_perm, &np, sizeof(int), cudaMemcpyHostToDevice);

	count = get_empty_indices(sudoku, empty);
;
	for (int i = 0; i < count; i++) {


		cudaMemcpyAsync(old_perm, new_perm, sizeof(int), cudaMemcpyDeviceToDevice);

		cudaMemset(new_perm, 0, sizeof(int));

		if (~i % 2) {
			generate_next_permutations << <GRID_DIMENSION, BLOCK_DIMENSION >> > (n, p, old_perm, new_perm, empty[i]);
		}
		else {
			generate_next_permutations << <GRID_DIMENSION, BLOCK_DIMENSION >> > (p, n, old_perm, new_perm, empty[i]);
		}
		err();
	}
	if (count % 2) {
		cudaMemcpy(result, p, 81 * sizeof(char), cudaMemcpyDeviceToHost);
	}
	else {
		cudaMemcpy(result, n, 81 * sizeof(char), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time used: %f ms\n", milliseconds);
	cudaFree(n);
	cudaFree(p);

	cudaFree(old_perm);
	cudaFree(new_perm);


	return result;

}


