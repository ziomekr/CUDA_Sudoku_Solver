
#include "sudoku_CPU.h"
#include <stdio.h>
#include <iostream>
#include <bitset>
#include <cmath>
#include <ctime>

#define DIMENSION 9	
#define SUBSQUARE_DIMENSION 3
#define EMPTY 0
#define SUDOKU_SOLVED -1


std::bitset<DIMENSION + 1> row_used_numbers(char* sudoku_board, int row) {
	std::bitset<DIMENSION + 1> used_numbers;
	for (int i = 0; i < DIMENSION; i++) {
		used_numbers[sudoku_board[row+i]] = 1;
	}
	return used_numbers;
}

std::bitset<DIMENSION + 1> column_used_numbers(char* sudoku_board, int column) {
	std::bitset<DIMENSION + 1> used_numbers;
	for (int i = 0; i < DIMENSION; i++) {
		used_numbers[sudoku_board[column + i * DIMENSION]] = 1;
	}
	return used_numbers;
}

std::bitset<DIMENSION + 1> subsquare_used_numbers(char* sudoku_board, int subsquare_top_left_cell) {
	std::bitset<DIMENSION + 1> used_numbers;
	for (int i = subsquare_top_left_cell; i < subsquare_top_left_cell + SUBSQUARE_DIMENSION; i++) {
		for (int j = 0; j < SUBSQUARE_DIMENSION; j++) {
			used_numbers[sudoku_board[i + j * DIMENSION]] = 1;
		}
	}
	return used_numbers;
}

std::bitset<DIMENSION + 1> get_valid_numbers(char* sudoku_board, int empty_cell) {
	std::bitset<DIMENSION + 1> used_numbers;
	int column = empty_cell % DIMENSION;
	int row = (empty_cell /  DIMENSION)*DIMENSION;
	int subsquare_top_left_cell = empty_cell - (empty_cell % SUBSQUARE_DIMENSION) - ((empty_cell / DIMENSION) % SUBSQUARE_DIMENSION)*DIMENSION;
	used_numbers = row_used_numbers(sudoku_board, row) | column_used_numbers(sudoku_board, column) | subsquare_used_numbers(sudoku_board, subsquare_top_left_cell);
	return used_numbers.flip();
}

int get_empty_cell(char* sudoku_board) {

	for (int empty_index = 0; empty_index < DIMENSION*DIMENSION; empty_index++) {
		if (sudoku_board[empty_index] == EMPTY)
			return empty_index;
	}
	int a = 0;
	return SUDOKU_SOLVED;
}

bool solve_sudokuCPU(char* sudoku_board) {

	int empty = get_empty_cell(sudoku_board);
	if (empty == SUDOKU_SOLVED)
		return true;
	
	else {
		std::bitset<DIMENSION + 1> valid_numbers = get_valid_numbers(sudoku_board, empty);
		for (int i = 1; i < DIMENSION + 1; i++) {
			if (valid_numbers[i] == 1) {
				sudoku_board[empty] = (char)i;
				if (solve_sudokuCPU(sudoku_board))
					return true;
			}
		}
		sudoku_board[empty] = EMPTY;
	}
	return false;
}


