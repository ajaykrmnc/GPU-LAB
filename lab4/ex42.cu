#include <stdio.h>
#include <time.h>
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
	return col + row * cols + threadIdx.x;
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

__global__ void matTransposePart(float *mat, float *res, int rows, int cols) {
	int index = index_calculate(blockIdx.x, blockIdx.y, gridDim.x);
	int transpose_index = index_calculate(blockIdx.y, blockIdx.x, gridDim.x);
	if (index < (rows * cols) && transpose_index < (rows * cols)) {
		res[transpose_index] = mat[index];
	}
}

void func(dim3 grid, dim3 block, int rows, int cols) {
  float *mata, *matb, *transpose_mata, *transpose_matb, *res;
	float *dev_mata, *dev_matb, *dev_transpose_mata, *dev_transpose_matb, *dev_res;	
	int mat_size = rows * cols * sizeof(float);

	mata = (float*)malloc(mat_size);
	matb = (float*)malloc(mat_size);
	transpose_mata = (float*)malloc(mat_size);
	transpose_matb = (float*)malloc(mat_size);
	res = (float*)malloc(mat_size);

	init_data(mata, rows, cols);
	init_data(matb, rows, cols);

	cudaMalloc((void **)&dev_mata, mat_size);
	cudaMalloc((void **)&dev_matb, mat_size);
	cudaMalloc((void **)&dev_transpose_mata, mat_size);
	cudaMalloc((void **)&dev_transpose_matb, mat_size);
	cudaMalloc((void **)&dev_res, mat_size);

	cudaMemcpy(dev_mata, mata, mat_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matb, matb, mat_size, cudaMemcpyHostToDevice);

	matTransposePart<<<grid, block>>>(dev_mata, dev_transpose_mata, rows, cols);
	cudaDeviceSynchronize();
	cudaMemcpy(transpose_mata, dev_transpose_mata, mat_size, cudaMemcpyDeviceToHost);

	matTransposePart<<<grid, block>>>(dev_matb, dev_transpose_matb, rows, cols);
	cudaDeviceSynchronize();
	cudaMemcpy(transpose_matb, dev_transpose_matb, mat_size, cudaMemcpyDeviceToHost);

	matMulPart<<<grid, block>>>(dev_mata, dev_matb, dev_res, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(res, dev_res, mat_size, cudaMemcpyDeviceToHost);

	printf("MatA * MatB: \n");
	print_mat(res, rows, cols);

	matMulPart<<<grid, block>>>(dev_transpose_mata, dev_transpose_matb, dev_res, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(res, dev_res, mat_size, cudaMemcpyDeviceToHost);

	printf("Transpose MatA * Transpose MatB: \n");
	print_mat(res, rows, cols);

	free(mata);
	free(matb);
	free(transpose_mata);
	free(transpose_matb);
	free(res);

	cudaFree(dev_mata);
	cudaFree(dev_matb);
	cudaFree(dev_transpose_mata);
	cudaFree(dev_transpose_matb);
	cudaFree(dev_res);

	cudaDeviceReset();

}

double exec_time(dim3 grid, dim3 block, int rows, int cols) {
	clock_t begin = clock();
	func(grid, block, rows, cols);
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}

int main() {
	// Using small rows and cols since it is difficult to get the output for large matrix
	int rows = N;
	int cols = N;

	dim3 grid1(1, 1, 1);
	dim3 block1(rows * cols);
	double t1 = exec_time(grid1, block1, rows, cols);
  printf("\n");

	dim3 grid2(cols, rows, 1);
	dim3 block2(1);
	double t2 = exec_time(grid2, block2, rows, cols);
  printf("\n");

	printf("Execution Time (Just Threads): %lf\n\n", t1);
	printf("Execution Time (Just Blocks): %lf\n\n", t2);

	return 0;
}

