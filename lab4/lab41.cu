#include <stdio.h>
#include <cuda_runtime.h>

#define N 3

void init_data(float *mat, int rows, int cols) {
	for(int row = 0; row < rows; row++) {
		for(int col = 0; col < cols; col++) {
			int index = row * cols + col;
			mat[index] = index;
		}
	}
}

void print_mat(float *mat, int rows, int cols) {
	for(int row = 0; row < rows; row++) {
		for(int col = 0; col < cols; col++) {
			int index = row * cols + col;
			printf("%f ", mat[index]);
		}
		printf("\n");
	}
}

__device__ int index_calculate(int row, int col, int cols) {
	return col + row * cols;
}

__global__ void matMulPart(float *mat1, float *mat2, float *res, int rows, int cols) {
	int index = index_calculate(blockIdx.x, blockIdx.y, gridDim.x);
	if (index < (rows * cols)) {
		float temp = 0;
		for(int i = 0; i < cols; i++) {
			temp += mat1[index_calculate(i, blockIdx.y, rows)] * mat2[index_calculate(blockIdx.x, i, cols)];
		}
		res[index] = temp;
	}
}

int main() {
	float *mata, *matb, *matc, *res;
	float *dev_mata, *dev_matb, *dev_matc, *dev_res;	
	int rows = N;
	int cols = N;
	int mat_size = rows * cols * sizeof(float);

	dim3 grid(cols, rows, 1);
	dim3 block(1);

	mata = (float*)malloc(mat_size);
	matb = (float*)malloc(mat_size);
	matc = (float*)malloc(mat_size);
	res = (float*)malloc(mat_size);

	init_data(mata, rows, cols);
	init_data(matb, rows, cols);
	init_data(matc, rows, cols);

	cudaMalloc((void **)&dev_mata, mat_size);
	cudaMalloc((void **)&dev_matb, mat_size);
	cudaMalloc((void **)&dev_matc, mat_size);
	cudaMalloc((void **)&dev_res, mat_size);

	cudaMemcpy(dev_mata, mata, mat_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matb, matb, mat_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matc, matc, mat_size, cudaMemcpyHostToDevice);

	matMulPart<<<grid, block>>>(dev_mata, dev_matb, dev_res, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(dev_matb, dev_res, mat_size, cudaMemcpyDeviceToDevice);

	matMulPart<<<grid, block>>>(dev_matb, dev_matc, dev_res, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(res, dev_res, mat_size, cudaMemcpyDeviceToHost);

	printf("Ans: \n");
	print_mat(res, rows, cols);

	free(mata);
	free(matb);
	free(matc);
	free(res);

	cudaFree(dev_mata);
	cudaFree(dev_matb);
	cudaFree(dev_matc);
	cudaFree(dev_res);

	cudaDeviceReset();

	return 0;
}
