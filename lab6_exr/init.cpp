#include "mat_mul.h"
#include "stdio.h"
#include "stdlib.h"

#define N 8

void initialData(float *ip, const int size) {
  for (int i = 0; i < size; i++) {
    ip[i] = ((float)rand() / (float)(RAND_MAX));
  }
}

void printMat(float *mat, int rows, int cols) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < cols; col++) {
      int index = row * cols + col;
      printf("%f ", mat[index]);
    }
    printf("\n");
  }
}

int main() {
  int width = N;
  int rows = N;
  int cols = N;
  int mat_bytes = rows * cols * sizeof(float);

  float *mat1, *mat2, *mat3;
  mat1 = (float *)malloc(mat_bytes);
  mat2 = (float *)malloc(mat_bytes);
  mat3 = (float *)malloc(mat_bytes);

  initialData(mat1, rows * cols);
  initialData(mat2, rows * cols);

  matMul(mat1, mat2, mat3, rows, cols);

  printf("Mat 1 = \n");
  printMat(mat1, rows, cols);
  printf("Mat 2 = \n");
  printMat(mat2, rows, cols);
  printf("Non-Tiled Mat1 * Mat2 = \n");
  printMat(mat3, rows, cols);

  tiledMatMul(mat1, mat2, mat3, rows, cols, width);
  printf("Tiled Mat1 * Mat2 = \n");
  printMat(mat3, rows, cols);

  free(mat1);
  free(mat2);
  free(mat3);

  return 0;
}
