#pragma once

__device__ int row_used_numbers(char* sudokus_arr, unsigned long long row, int* masks);

__device__ int column_used_numbers(char* sudokus_arr, unsigned long long column, int* masks);

__device__ int subsquare_used_numbers(char* sudokus_arr, unsigned long long subsquare_top_left_cell, int* masks);

__device__ int get_valid_numbers(char* sudokus_arr, unsigned long long cell, unsigned long long top_left_sudoku_cell, int* masks);

__device__ void copy_sudoku(char* src_sudoku, char* dest_sudoku);

__device__ void fill_masks(int * masks);

__global__ void generate_next_permutations(char* sudokus_arr, char* sudoku_arr_new_permutations, unsigned long long* number_of_old_permutations, unsigned long long* number_of_permutations, int empty_cell, unsigned long long max_permutations, bool* max_permutations_overflow);

__host__ __device__ void printT(char * sudoku);

__global__ void backtrackigKernel(char* sudokus_arr, int number_of_permutations, unsigned long long* current_sudoku_index, int* empty_cells, int empty_cells_count, bool* solved);

__host__ size_t get_free_memory_size();

__host__ size_t calculate_max_number_of_permutations();

__host__ void free_memory_GPU(char * sudokus_arr1, char * sudokus_arr2, unsigned long long* number_of_permutations1, unsigned long long* number_of_permutations2, int * empty_cells);

__host__ void err();

__host__ int get_empty_indices(char* sudoku, int* empty);

__host__ char* solve_sudokuGPU(char*  sudoku);
