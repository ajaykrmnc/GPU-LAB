#include "AddMul.h"
#include "cuda_runtime.h"
#include "stdio.h"

__device__ int index_calculate(int row, int col, int cols) {
  return col + row * cols;
}

__global__ void matAddPart(float *mat1, float *mat2, float *res, int rows,
                           int cols) {
  int index = index_calculate(blockIdx.x, blockIdx.y, gridDim.x);
  if (index < (rows * cols)) {
    res[index] = mat1[index] + mat2[index];
  }
}

__global__ void matMulPart(float *mat1, float *mat2, float *res, int rows,
                           int cols) {
  int index = index_calculate(blockIdx.x, blockIdx.y, gridDim.x);
  if (index < (rows * cols)) {
    float temp = 0;
    for (int i = 0; i < cols; i++) {
      temp += mat1[index_calculate(i, blockIdx.y, rows)] *
              mat2[index_calculate(blockIdx.x, i, cols)];
    }
    res[index] = temp;
  }
}

void matrixAdd(float *mat1, float *mat2, float *res, int rows, int cols) {
  int mat_bytes = rows * cols * sizeof(float);
  dim3 grid(cols, rows, 1);
  dim3 block(1);

  float *dev_mat1, *dev_mat2, *dev_res;

  cudaMalloc((void **)&dev_mat1, mat_bytes);
  cudaMalloc((void **)&dev_mat2, mat_bytes);
  cudaMalloc((void **)&dev_res, mat_bytes);

  cudaMemcpy(dev_mat1, mat1, mat_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mat2, mat2, mat_bytes, cudaMemcpyHostToDevice);

  matAddPart<<<grid, block>>>(dev_mat1, dev_mat2, dev_res, rows, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(res, dev_res, mat_bytes, cudaMemcpyDeviceToHost);

  cudaFree(dev_mat1);
  cudaFree(dev_mat2);
  cudaFree(dev_res);

  cudaDeviceReset();
}

void matrixMul(float *mat1, float *mat2, float *res, int rows, int cols) {
  int mat_bytes = rows * cols * sizeof(float);
  dim3 grid(cols, rows, 1);
  dim3 block(1);

  float *dev_mat1, *dev_mat2, *dev_res;

  cudaMalloc((void **)&dev_mat1, mat_bytes);
  cudaMalloc((void **)&dev_mat2, mat_bytes);
  cudaMalloc((void **)&dev_res, mat_bytes);

  cudaMemcpy(dev_mat1, mat1, mat_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_mat2, mat2, mat_bytes, cudaMemcpyHostToDevice);

  matMulPart<<<grid, block>>>(dev_mat1, dev_mat2, dev_res, rows, cols);
  cudaDeviceSynchronize();

  cudaMemcpy(res, dev_res, mat_bytes, cudaMemcpyDeviceToHost);

  cudaFree(dev_mat1);
  cudaFree(dev_mat2);
  cudaFree(dev_res);

  cudaDeviceReset();
}

