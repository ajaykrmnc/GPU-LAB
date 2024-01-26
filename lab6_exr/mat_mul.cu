#include<cuda_runtime.h>
#include<stdio.h>
#include"mat_mul.h"

__global__ void tiledMatrixMulKernel(float *mat1, float *mat2, float *mat3, int width) {
	__shared__ float mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float pvalue = 0;
	for(int ph = 0; ph < width / TILE_WIDTH; ph++) {
		mds[ty][tx] = mat1[row * width + ph * TILE_WIDTH + tx];
		nds[ty][tx] = mat2[(ph * TILE_WIDTH + ty) * width + col];
		__syncthreads();

		for(int k = 0; k < TILE_WIDTH; k++) {
			pvalue += mds[ty][k] * nds[k][tx];
		}
		__syncthreads();
	}
	mat3[row * width + col] = pvalue;
}

__device__ int index_calculate(int col, int row, int cols) {
	return col + row * cols;
}

__global__ void matrixMulKernel(float *mat1, float *mat2, float *res, int rows, int cols) {
	int index = index_calculate(blockIdx.x, blockIdx.y, gridDim.x);
	if (index < (rows * cols)) {
		float temp = 0;
		for(int i = 0; i < cols; i++) {
			temp += (mat1[index_calculate(i, blockIdx.y, rows)] * mat2[index_calculate(blockIdx.x, i, cols)]);
		}
		res[index] = temp;
	}
}

void tiledMatMul(float *mat1, float *mat2, float *mat3, int rows, int cols, int width) {
	int mat_bytes = rows * cols * sizeof(float);

	float *dev_mat1, *dev_mat2, *dev_mat3;
	cudaMalloc((void **)&dev_mat1, mat_bytes);
	cudaMalloc((void **)&dev_mat2, mat_bytes);
	cudaMalloc((void **)&dev_mat3, mat_bytes);

	cudaMemcpy(dev_mat1, mat1, mat_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, mat2, mat_bytes, cudaMemcpyHostToDevice);

	dim3 block(TILE_WIDTH, TILE_WIDTH);
	dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y, 1);

	tiledMatrixMulKernel<<<grid, block>>>(dev_mat1, dev_mat2, dev_mat3, width);
	cudaDeviceSynchronize();

	cudaMemcpy(mat3, dev_mat3, mat_bytes, cudaMemcpyDeviceToHost);
	
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);
	cudaFree(dev_mat3);

	cudaDeviceReset();
}

void matMul(float *mat1, float *mat2, float *mat3, int rows, int cols) {
	int mat_bytes = rows * cols * sizeof(float);

	float *dev_mat1, *dev_mat2, *dev_mat3;
	cudaMalloc((void **)&dev_mat1, mat_bytes);
	cudaMalloc((void **)&dev_mat2, mat_bytes);
	cudaMalloc((void **)&dev_mat3, mat_bytes);

	cudaMemcpy(dev_mat1, mat1, mat_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, mat2, mat_bytes, cudaMemcpyHostToDevice);

	dim3 grid(rows, cols, 1);
	dim3 block(1, 1, 1);

	matrixMulKernel<<<grid, block>>>(dev_mat1, dev_mat2, dev_mat3, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(mat3, dev_mat3, mat_bytes, cudaMemcpyDeviceToHost);

	cudaFree(dev_mat1);
	cudaFree(dev_mat2);
	cudaFree(dev_mat3);

	cudaDeviceReset();
}
