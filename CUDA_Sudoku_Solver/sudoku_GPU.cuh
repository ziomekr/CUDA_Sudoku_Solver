#pragma once

__device__ int row_used_numbers(char* sudokus_arr, int row, int* masks);

__device__ int column_used_numbers(char* sudokus_arr, int column, int* masks);

__device__ int subsquare_used_numbers(char* sudokus_arr, int subsquare_top_left_cell, int* masks);

__device__ int get_valid_numbers(char* sudokus_arr, int cell, int tIdx, int* masks);

__device__ void copy_sudoku(char* src_sudoku, char* dest_sudoku);

__device__ void fill_masks(int * masks);

__global__ void generate_next_permutations(char* sudokus_arr, char* sudoku_arr_new_permutations, int* number_of_old_permutations, int* number_of_permutations, int empty_cell);

__device__ void printT(char * sudoku);

__global__ void backtrackigKernel(char* sudokus_arr, int number_of_permutations, int* current_sudoku_index, int* empty_cells, int empty_cells_count, bool solved);

__host__ void err();

__host__ int get_empty_indices(char* sudoku, int* empty);

__host__ char* solve_sudokuGPU(char*  sudoku);
