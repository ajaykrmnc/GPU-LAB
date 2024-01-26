#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>

#define N 1024

__global__ void Vec_Per_Dist(int *vec_1, int *vec_2, int *res) {
	int index = blockIdx.x;
	if (index < N) {
		int temp = vec_1[index] - vec_2[index];
		res[index] = temp * temp;
	}
}

__global__ void Vec_Per_Norm(int *vec, int *res) {
	int index = blockIdx.x;
	if (index < N) {
		res[index] = vec[index] * vec[index];
	}
}

double sum_sqrt(int vec[]) {
	long dist = 0;
	for (int i = 0; i < N; i++) {
		dist += vec[i];
	}
	return sqrt(dist);

}

int main() {
	int vec_1[N], vec_2[N], res[N];
	int vec_size = N * sizeof(int);
	int *dev_vec_1, *dev_vec_2, *dev_res;
	
	// initialize vector 1 and vector 2
	for (int i = 0; i < N; i++) {
		vec_1[i] = (i + 1) * (i + 1);
		vec_2[i] = 2 * (i + 1) + 1;
	}
	
	// Allocate
	cudaMalloc((void **)&dev_vec_1, vec_size);
	cudaMalloc((void **)&dev_vec_2, vec_size);
	cudaMalloc((void **)&dev_res, vec_size);

	// Copu Host To Device
	cudaMemcpy(dev_vec_1, vec_1, vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec_2, vec_2, vec_size, cudaMemcpyHostToDevice);

	Vec_Per_Dist<<<N, 1>>>(dev_vec_1, dev_vec_2, dev_res);
	cudaMemcpy(res, dev_res, vec_size, cudaMemcpyDeviceToHost);
	double distance = sum_sqrt(res);
	printf("Distance: %lf\n", distance);

	Vec_Per_Norm<<<N, 1>>>(dev_vec_1, dev_res);
	cudaMemcpy(res, dev_res, vec_size, cudaMemcpyDeviceToHost);
	double norm_1 = sum_sqrt(res);
	printf("Vec X Euclidean Normal: %lf\n", norm_1);

	Vec_Per_Norm<<<N, 1>>>(dev_vec_2, dev_res);
	cudaMemcpy(res, dev_res, vec_size, cudaMemcpyDeviceToHost);
	double norm_2 = sum_sqrt(res);
	printf("Vec Y Euclidean Normal: %lf\n", norm_2);

	// Deallocate
	cudaFree(dev_vec_1);
	cudaFree(dev_vec_2);
	cudaFree(dev_res);

	return 0;
}


