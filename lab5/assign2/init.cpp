#include "AddMul.h"
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
  float *mat1, *mat2, *res;
  int rows = N;
  int cols = N;
  int mat_bytes = rows * cols * sizeof(float);

  mat1 = (float *)malloc(mat_bytes);
  mat2 = (float *)malloc(mat_bytes);
  res = (float *)malloc(mat_bytes);

  init_data(mat1, rows, cols);
  init_data(mat2, rows, cols);

  matrixAdd(mat1, mat2, res, rows, cols);

  printf("Mat1 = \n");
  print_mat(mat1, rows, cols);
  printf("\n");
  printf("Mat2 = \n");
  print_mat(mat2, rows, cols);
  printf("\n");
  printf("Mat1 + Mat2 = \n");
  print_mat(res, rows, cols);
  printf("\n");

  memcpy(mat1, res, mat_bytes);
  matrixMul(mat1, mat1, res, rows, cols);

  printf("Square(Mat1 + Mat2) = \n");
  print_mat(res, rows, cols);
  printf("\n");

  free(mat1);
  free(mat2);
  free(res);

  return 0;
}
