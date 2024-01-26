#include<cuda_runtime.h>
#include<stdio.h>
int main(int argc, char **argv)
{
int nElem = 1024;
dim3 block(1024);
dim3 grid ((nElem + block.x - 1)/block.x);
printf("grid.x %d block.x %d \n", grid.x, block.x);
//reset bloclk 
block.x = 512;
grid.x = (nElem + block.x - 1)/block.x;
printf("grid.x %d block.x %d \n", grid.x, block.x);

// reset block 
block.x = 256;
grid.x = (nElem + block.x - 1)/block.x;
printf("grid.x %d block.x %d\n", grid.x, block.x);

// reset block
block.x = 128;
grid. x = (nElem + block.x - 1)/block.x;
printf("grid.x %d block.x %d\n", grid.x, block.x);

// reset block 
block.x = 64;
grid.x = (nElem + block.x - 1)/block.x;
printf("grid.x %d block.x %d\n", grid.x, block.x);

// reset block 
block.x = 32;
grid.x = (nElem + block.x - 1)/block.x;
printf("grid.x %d block.x %d\n", grid.x, block.x);

// reset device before you leave 
cudaDeviceReset();
return(0);
}



