#include<cuda_runtime.h>
#include<stdio.h>
int main(int argc,char **argv)
{
int dev = 0;
cudaDeviceProp deviceProp;
cudaGetDeviceProperties(&deviceProp, dev);
printf("Warp Size: %d \n", deviceProp.warpSize);
printf("Maximum number of threads per multiprocesser: %d \n", deviceProp.maxThreadsPerMultiProcessor);
printf("Maximum number of threads per block : %d \n", deviceProp.maxThreadsPerBlock);
printf("maximum size of each dimension : %d x %d x %d \n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
printf("maximum size of grid : %d x %d x %d \n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
printf("maximum memory pitch : %lu bytes \n", deviceProp.memPitch);
exit(EXIT_SUCCESS);
}

