
#include<stdio.h>
__global__ void helloFromGPU()
{
   printf("Hello World From GPU\n");
}
int main(int  argc,char **argv)
{
   printf("Hello World From GPU\n");
   helloFromGPU<<<1 , 5 >>>();
   cudaDeviceReset();
    return 0;
}


