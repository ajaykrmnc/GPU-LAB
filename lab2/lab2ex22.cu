#include<cuda_runtime.h>
#include<stdio.h>
#include<math.h>

#define N 1024

__global__ void Standard_Deviation_Part(int *vec, float *mean, float *res) {
	int index = blockIdx.x;
	if(index < N) {
		float temp = (float)vec[index] - *mean;
		res[index] = temp * temp;
	}
}

int main() {
	int vec[N];
	float res[N];
	float mean = 0.0;
	int vec_size = N * sizeof(int);
	int res_size = N * sizeof(float);

	// GPU varraibles 
	int *gpu_vec;
	float *gpu_res;
	float *gpu_mean;

	// Allocate GPU Memory 
	cudaMalloc((void **)&gpu_vec, vec_size);
	cudaMalloc((void **)&gpu_res, res_size);
	cudaMalloc((void **)&gpu_mean, sizeof(float));

	for (int i = 0; i < N; ++i) {
		vec[i] = 2 * (i + 1) + 1;
		mean += vec[i];
	}

	mean/=N;
	
	// Copy Host To Device 
	cudaMemcpy(gpu_vec, vec, vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_mean, &mean, sizeof(float), cudaMemcpyHostToDevice);

	Standard_Deviation_Part<<<N, 1>>>(gpu_vec, gpu_mean, gpu_res);

	// Copy Device To Host
	cudaMemcpy(res, gpu_res, res_size, cudaMemcpyDeviceToHost);

	float sd = 0.0;
	for (int i = 0; i < N; i++) {
		sd += res[i];
	}

	sd /= N;

	sd = sqrt(sd);

	printf("Standard Deviation : %f\n", sd);

	// Free Memory 
	cudaFree(gpu_vec);
	cudaFree(gpu_res);
	cudaFree(gpu_mean);

	return 0;
}


