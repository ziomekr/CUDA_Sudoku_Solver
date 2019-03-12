#pragma once
#include <bitset>

#define DIMENSION 9

std::bitset<DIMENSION + 1> row_used_numbers(char* sudoku_board, int row);
std::bitset<DIMENSION + 1> column_used_numbers(char* sudoku_board, int column);
std::bitset<DIMENSION + 1> subsquare_used_numbers(char* sudoku_board, int subsquare_top_left_cell);

std::bitset<DIMENSION + 1> get_valid_numbers(char * sudoku_board, int empty_cell);

int get_empty_cell(char * sudoku_board);

bool solve_sudokuCPU(char * sudoku_board);
