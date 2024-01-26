#ifndef MAT_MUL_H
#define MAT_MUL_H

#define TILE_WIDTH 2

void matMul(float *mat1, float *mat2, float *mat3, int rows, int cols);

void tiledMatMul(float *mat1, float *mat2, float *mat3, int rows, int cols, int width);

#endif
