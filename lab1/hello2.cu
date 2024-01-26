#include<stdio.h>
__global__ void helloFromGPU()
{
printf("GPU Computing  \t\t\tProgrsm:Hello World, a Kernel Call and Passing Parameters\t\t\t10-08-2022\n");
}
void helloFromCPU()
{
printf("GPU Computing \t\t\tProgram: Hello World , a Kernel Call and Passing Parameters\t\t\t10-08-2022\n");
}
int main(int argc,char **argv)
{
  helloFromGPU<<<1,4>>>();
  for(int i=0;i<4;i++)
{
helloFromCPU();
}
cudaDeviceReset();
return 0;
}
