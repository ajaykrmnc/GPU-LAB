#include "Helper.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#define N 3

void init_data(float *mat, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      int index = row * cols + col;
      mat[index] = index;
    }
  }
}

void print_mat(float *mat, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      int index = row * cols + col;
      printf("%f ", mat[index]);
    }
    printf("\n");
  }
}

int main() {
  float *mat, *tran_mat, *res;
  int rows = N;
  int cols = N;
  int mat_bytes = rows * cols * sizeof(float);

  mat = (float *)malloc(mat_bytes);
  tran_mat = (float *)malloc(mat_bytes);
  res = (float *)malloc(mat_bytes);

  init_data(mat, rows, cols);

  matrixTranspose(mat, tran_mat, rows, cols);

  printf("Mat = \n");
  print_mat(mat, rows, cols);
  printf("\n");
  printf("Transpose(Mat) = \n");
  print_mat(tran_mat, rows, cols);
  printf("\n");

  matrixMul(mat, tran_mat, res, rows, cols);
  printf("Mat * Transpose(Mat) = \n");
  print_mat(res, rows, cols);
  printf("\n");

  free(mat);
  free(tran_mat);
  free(res);

  return 0;
}
