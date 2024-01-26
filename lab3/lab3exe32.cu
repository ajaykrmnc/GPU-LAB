#include <stdio.h>
#include<time.h>
#include <cuda_runtime.h>

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

__global__ void matAddPart(float *mat1, float *mat2, float *res, int rows, int cols) {
	int index = blockIdx.x + blockIdx.y * gridDim.x + threadIdx.x;
	if (index < (rows * cols)) {
		res[index] = mat1[index] + mat2[index];
	}
}

void sumMat(dim3 grid, dim3 block, int rows, int cols) {
	float *mat1, *mat2, *res, *dev_mat1, *dev_mat2, *dev_res;
	int mat_bytes = rows * cols * sizeof(float);

	mat1 = (float *)malloc(mat_bytes);
	mat2 = (float *)malloc(mat_bytes);
	res = (float *)malloc(mat_bytes);

	init_data(mat1, rows, cols);
	init_data(mat2, rows, cols);

	cudaMalloc((void **)&dev_mat1, mat_bytes);
	cudaMalloc((void **)&dev_mat2, mat_bytes);
	cudaMalloc((void **)&dev_res, mat_bytes);

	cudaMemcpy(dev_mat1, mat1, mat_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, mat2, mat_bytes, cudaMemcpyHostToDevice);

	matAddPart<<<grid, block>>>(dev_mat1, dev_mat2, dev_res, rows, cols);
	cudaDeviceSynchronize();

	cudaMemcpy(res, dev_res, mat_bytes, cudaMemcpyDeviceToHost);

	printf("Ans: \n");
	print_mat(res, rows, cols);
	
	// Free Host mem
	free(mat1);
	free(mat2);
	free(res);

	// Free Device Mem
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);
	cudaFree(dev_res);

	cudaDeviceReset();

}

double exec_time(dim3 grid, dim3 block, int rows, int cols) {
	clock_t begin = clock();
	sumMat(grid, block, rows, cols);
	clock_t end = clock();
	return (double)(end - begin) / CLOCKS_PER_SEC;
}

int main() {
  // Using small rows and cols since it is difficult to get the output for large matrix
	int rows = 4;
	int cols = 5;

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

  // It seems using Just Blocks with a single thread is wasteful and slower than using a single block with multiple threads for significantly large inputs (tested with 400 * 500 matrix)
  // From this we can conclude that a combination of threads and blocks should be used

	return 0;
}

