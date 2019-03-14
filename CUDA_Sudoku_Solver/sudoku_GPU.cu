#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"
#include "stdlib.h"
#include "sudoku_GPU.cuh"


#define DIMENSION 16
#define SUBSQUARE_DIMENSION 4
#define BLOCK_DIMENSION 256
#define GRID_DIMENSION 512
#define MAX_SUDOKUS_COUNT 5000000


__device__
int row_used_numbers(char* sudokus_arr, unsigned long long row, int* masks) {
	int used_numbers = 0;
	for (unsigned long long i = row; i < row + DIMENSION; i++) {
		used_numbers = used_numbers | masks[sudokus_arr[i]];
	}
	return used_numbers;
}

__device__
int column_used_numbers(char* sudokus_arr, unsigned long long column, int* masks) {
	int used_numbers = 0;
	for (unsigned long long i = 0; i < DIMENSION; i++) {
		used_numbers = used_numbers | masks[sudokus_arr[column + i * DIMENSION]];
	}
	return used_numbers;
}

__device__
int subsquare_used_numbers(char* sudokus_arr, unsigned long long subsquare_top_left_cell, int* masks) {
	int used_numbers = 0;
	for (unsigned long long i = subsquare_top_left_cell; i < subsquare_top_left_cell + SUBSQUARE_DIMENSION; i++) {
		for (unsigned long long j = 0; j < SUBSQUARE_DIMENSION; j++) {
			used_numbers = used_numbers | masks[sudokus_arr[i + j * DIMENSION]];
		}
	}
	return used_numbers;
}

__device__
int get_valid_numbers(char* sudokus_arr, unsigned long long cell, unsigned long long top_left_sudoku_cell, int* masks) {
	unsigned long long board_offset = cell % DIMENSION;
	unsigned long long row = cell - board_offset;
	unsigned long long column = top_left_sudoku_cell + board_offset;
	unsigned long long subsquare_top_left_cell = cell - (cell % SUBSQUARE_DIMENSION) - (((cell - top_left_sudoku_cell) / DIMENSION) % SUBSQUARE_DIMENSION)*DIMENSION;
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
void generate_next_permutations(char* sudokus_arr, char* sudoku_arr_new_permutations, unsigned long long* number_of_old_permutations, unsigned long long* number_of_permutations, int empty_cell, unsigned long long max_permutations, bool* solved){

	
	unsigned long long tIdx = threadIdx.x + blockDim.x*blockIdx.x;
	unsigned long long id = 0;
	unsigned long long new_permutation_id = 0;
	__shared__ int masks[DIMENSION + 1];
	__shared__ int valid_numbers[BLOCK_DIMENSION];
	
	fill_masks(masks);


	while (tIdx < *number_of_old_permutations && tIdx < max_permutations) {
		
		unsigned long long top_left_cell = tIdx * DIMENSION * DIMENSION;
		
		valid_numbers[threadIdx.x] = get_valid_numbers(sudokus_arr, top_left_cell + (unsigned long long)empty_cell, top_left_cell, masks);
		
		
		for (int i = 1; i < DIMENSION + 1; i++) {
			if ((masks[i] & (~valid_numbers[threadIdx.x])) != 0) {
				id = atomicAdd(number_of_permutations, 1);
				new_permutation_id = id * DIMENSION*DIMENSION;
				if (id < max_permutations) {
						copy_sudoku(sudokus_arr + top_left_cell, sudoku_arr_new_permutations + new_permutation_id);
						*(sudoku_arr_new_permutations + new_permutation_id + empty_cell) = (char)i;
				}
				else {
					
					return;
				}
			}
		}
		tIdx += blockDim.x * gridDim.x;
	}		
}

__device__ void printT(char* sudoku) {
	printf("-----------------------------------------------------------------------\n");
	for (int i = 0; i < DIMENSION*DIMENSION; i++) {
		printf("%d ", sudoku[i]);
		if (i%DIMENSION == DIMENSION - 1)
			printf("\n");
	}
	printf("-----------------------------------------------------------------------\n");
}

__global__ void backtrackigKernel(char* sudokus_arr, int number_of_permutations, int* current_sudoku_index, int* empty_cells, int empty_cells_count, bool solved)
{
	__shared__ int masks[DIMENSION + 1];
	__shared__ int valid_numbers[BLOCK_DIMENSION];
	__shared__ int empty_cells_offsets[BLOCK_DIMENSION];
	fill_masks(masks);
	int sudoku_index = atomicAdd(current_sudoku_index, 1);
	int sudokus_arr_idx = sudoku_index * DIMENSION * DIMENSION;
	while ((sudoku_index < number_of_permutations) && !solved) {
		
		valid_numbers[threadIdx.x] = get_valid_numbers(sudokus_arr, sudokus_arr_idx + *(empty_cells + empty_cells_offsets[threadIdx.x]), sudokus_arr_idx, masks);
		
		empty_cells_offsets[threadIdx.x] = 0;
		
		for (int i = 1; i < DIMENSION + 1; i++) {
		
			if ((masks[i] & (~valid_numbers[threadIdx.x])) != 0) {
				
				sudokus_arr[sudokus_arr_idx + *(empty_cells + empty_cells_offsets[threadIdx.x])] = (char)i;
				empty_cells_offsets[threadIdx.x] += 1;
				
				if (empty_cells_offsets[threadIdx.x] < empty_cells_count) {					
					i = 0;					
					valid_numbers[threadIdx.x] = get_valid_numbers(sudokus_arr, sudokus_arr_idx + *(empty_cells + empty_cells_offsets[threadIdx.x]), sudokus_arr_idx, masks);
				}
				else {
					solved = true;
					break;
				}
			}
			else {
				while (i == DIMENSION) {
					sudokus_arr[sudokus_arr_idx + *(empty_cells + empty_cells_offsets[threadIdx.x])] = 0;
					empty_cells_offsets[threadIdx.x] -= 1;
					i = sudokus_arr[sudokus_arr_idx + *(empty_cells + empty_cells_offsets[threadIdx.x])];
					sudokus_arr[sudokus_arr_idx + *(empty_cells + empty_cells_offsets[threadIdx.x])] = 0;
				}
				if (empty_cells_offsets[threadIdx.x] > -1) {

					valid_numbers[threadIdx.x] = get_valid_numbers(sudokus_arr, sudokus_arr_idx + *(empty_cells + empty_cells_offsets[threadIdx.x]), sudokus_arr_idx, masks);
				}
				else {
					sudoku_index = atomicAdd(current_sudoku_index, 1);
					break;
				}
			}
		}	
	}
}

__host__
int get_empty_indices(char* sudoku, int* empty) {
	int count = 0;
	for (int i = 0; i < DIMENSION*DIMENSION; i++) {
		if ((int)sudoku[i] == 0) {
			empty[count++] = i;
		}
	}
	return count;
}

__host__ 
size_t get_free_memory_size() {
	size_t free_memory;
	cudaMemGetInfo(&free_memory, nullptr);
	return free_memory;
}

__host__ 
unsigned long long calculate_max_number_of_permutations() {
	return (get_free_memory_size() / (2 * DIMENSION*DIMENSION * sizeof(char)) - 128);
}

__host__ void allocate_memory_GPU(char* sudokus_arr1, char* sudokus_arr2, int* number_of_permutations1, int* number_of_permutations2, int* empty_cells, int empty_cells_count, size_t max_permutations_number) {
	
}

__host__ 
void free_memory_GPU(char* sudokus_arr1, char* sudokus_arr2, int* number_of_permutations1, int* number_of_permutations2, int* empty_cells, int empty_cells_count, size_t max_permutations_number) {
	cudaFree(sudokus_arr1);
	cudaFree(sudokus_arr2);
	cudaFree(number_of_permutations1);
	cudaFree(number_of_permutations2);
	cudaFree(empty_cells);
}

__host__
void err() {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("CUDA error: %s\n", cudaGetErrorString(error));
	}	
}

__host__ 
void initializeGPU(char* sudoku, char* sudokus_arr1, char* sudokus_arr2, int* number_of_permutations1, int* number_of_permutations2, int* empty_cellsGPU, int* empty_cells, int empty_cells_count, size_t max_permutations_number) {
	allocate_memory_GPU(sudokus_arr1, sudokus_arr2, number_of_permutations1, number_of_permutations2, empty_cellsGPU, empty_cells_count, max_permutations_number);
	
}
__host__
char* solve_sudokuGPU(char* sudoku) {
	
	int empty_indices[DIMENSION*DIMENSION];
	int empty_count = get_empty_indices(sudoku, empty_indices);
	unsigned long long max_permutations_number;
	char result[DIMENSION*DIMENSION];
	unsigned long long c=0, d=0;

	char* sudokus_arr1, *sudokus_arr2;
	int* *empty_indicesGPU;
	unsigned long long *number_of_permutations1, *number_of_permutations2;
	bool* solvedGPU, *solved;

	cudaMalloc(&number_of_permutations1, sizeof(unsigned long long));
	cudaMalloc(&number_of_permutations2, sizeof(unsigned long long));
	cudaMalloc(&empty_indicesGPU, sizeof(int)*empty_count);
	cudaHostAlloc(&solved, sizeof(bool), cudaHostAllocMapped);
	cudaHostGetDevicePointer(&solvedGPU, solved, 0);

	max_permutations_number = calculate_max_number_of_permutations();
	printf("%zu\n", max_permutations_number);
	cudaMalloc(&sudokus_arr1, sizeof(char)*max_permutations_number*DIMENSION*DIMENSION);
	err();
	cudaMalloc(&sudokus_arr2, sizeof(char)*max_permutations_number*DIMENSION*DIMENSION);
	err();
	

	cudaMemcpy(sudokus_arr1, sudoku, DIMENSION * DIMENSION * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemset(number_of_permutations1, 1, 1);
	cudaMemset(number_of_permutations2, 0, sizeof(int));
	cudaMemcpy(empty_indicesGPU, empty_indices, empty_count * sizeof(int), cudaMemcpyHostToDevice);
	

	
	for (int i = 0; i < empty_count; i++) {
		if (i % 2) {
			generate_next_permutations << <GRID_DIMENSION, BLOCK_DIMENSION >> > (sudokus_arr1, sudokus_arr2, number_of_permutations1, number_of_permutations2, empty_indices[i], max_permutations_number, solvedGPU);
			
			cudaDeviceSynchronize();
			cudaMemcpy(&c, number_of_permutations2, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			cudaMemcpy(&d, number_of_permutations1, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

			cudaMemset(number_of_permutations1, 0, sizeof(unsigned long long));
		}
		else {
			generate_next_permutations << <GRID_DIMENSION, BLOCK_DIMENSION >> > (sudokus_arr2, sudokus_arr1, number_of_permutations2, number_of_permutations1, empty_indices[i], max_permutations_number, solvedGPU);
			
			cudaDeviceSynchronize();
			cudaMemcpy(&c, number_of_permutations1, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			cudaMemcpy(&d, number_of_permutations2, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

			cudaMemset(number_of_permutations2, 0, sizeof(unsigned long long));

		}
		err();
		printf("%zu %zu\n", c, d);

	}
	return result;
}
//
//__host__
//char* solve_sudokuGPU(char* sudoku) {
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	
//	int count;
//	int empty[DIMENSION*DIMENSION];
//	int *old_perm, *new_perm;
//	char *n, *p;
//	char *result;
//	result = (char*)malloc(sizeof(char) * 81);
//	int np = 1;
//	cudaEventRecord(start);
//	cudaMalloc((void**)&n, MAX_SUDOKUS_COUNT * DIMENSION * DIMENSION * sizeof(char));
//	err();
//	cudaMalloc((void**)&p, MAX_SUDOKUS_COUNT * DIMENSION * DIMENSION * sizeof(char));
//	err();
//
//	cudaMalloc((void**)&old_perm, sizeof(int));
//	
//	cudaMalloc((void**)&new_perm, sizeof(int));
//	cudaMemcpy(n, sudoku, DIMENSION * DIMENSION * sizeof(char), cudaMemcpyHostToDevice);
//	cudaMemcpy(new_perm, &np, sizeof(int), cudaMemcpyHostToDevice);
//
//	count = get_empty_indices(sudoku, empty);
//;
//	for (int i = 0; i < count; i++) {
//
//
//		cudaMemcpyAsync(old_perm, new_perm, sizeof(int), cudaMemcpyDeviceToDevice);
//
//		cudaMemset(new_perm, 0, sizeof(int));
//
//		if (~i % 2) {
//			generate_next_permutations << <GRID_DIMENSION, BLOCK_DIMENSION >> > (n, p, old_perm, new_perm, empty[i]);
//		}
//		else {
//			generate_next_permutations << <GRID_DIMENSION, BLOCK_DIMENSION >> > (p, n, old_perm, new_perm, empty[i]);
//		}
//		err();
//	}
//	if (count % 2) {
//		cudaMemcpy(result, p, 81 * sizeof(char), cudaMemcpyDeviceToHost);
//	}
//	else {
//		cudaMemcpy(result, n, 81 * sizeof(char), cudaMemcpyDeviceToHost);
//	}
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//
//	float milliseconds = 0;
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("GPU time used: %f ms\n", milliseconds);
//	cudaFree(n);
//	cudaFree(p);
//
//	cudaFree(old_perm);
//	cudaFree(new_perm);
//
//	int count = 0;
//	int empty[DIMENSION*DIMENSION];
//	char *n;
//	int* empty_c;
//	count = get_empty_indices(sudoku, empty);
//	char* result;
//	result = (char*)malloc(sizeof(char) * 81);
//	int* c_i;
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start);
//	cudaMalloc((void**)&n, DIMENSION * DIMENSION * sizeof(char)*10);
//	cudaMalloc((void**)&empty_c, count * sizeof(int));
//	cudaMalloc((void**)&c_i, sizeof(int));
//	
//	cudaMemcpy(n, sudoku, DIMENSION * DIMENSION * sizeof(char)*10, cudaMemcpyHostToDevice);
//	cudaMemcpy(empty_c, empty, count * sizeof(int), cudaMemcpyHostToDevice);
//	cudaMemset(c_i, 0, sizeof(int));
//	backtrackigKernel << < GRID_DIMENSION, BLOCK_DIMENSION >> > (n, 10, c_i, empty_c, count, false);
//	cudaDeviceSynchronize();
//	err();
//	err();
//	cudaMemcpy(result, n, 81 * sizeof(char), cudaMemcpyDeviceToHost);
//	err();
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	float milliseconds = 0;
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("GPU time used: %f ms\n", milliseconds);
//	cudaFree(n);
//	cudaFree(empty_c);
//	cudaFree(c_i);
//	return result;
//
//}


