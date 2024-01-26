
#include<stdio.h>
__global__ void helloFromGPU()
{
   printf("Shivam Singh from GPU\n");
}
 void helloFromCPU()
{
 printf("Shivam Singh from CPU\n");
}
 int main( int argc , char **argv)
{
 helloFromGPU<<<1, 10>>>();
 for(int i=0;i<10;i++)
{
 helloFromCPU();
}

 
 cudaDeviceReset();
 return 0;
}
