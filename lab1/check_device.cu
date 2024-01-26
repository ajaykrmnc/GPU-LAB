#include<cuda_runtime.h>
#include<stdio.h>
using namespace std;
int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if(deviceCount == 0)
    {
        printf("There are no available devices that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable devices \n", deviceCount);
    }
    int dev = 0, driverVersion = 0, runtimeVersion = 0;

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d \n", driverVersion / 100, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("CUDA Capability Major/Minor version number %d.%d \n", deviceProp.major, deviceProp.minor);
    printf("Total Amount of global Memory : %.02f GBytes (%llu " "bytes) \n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3), (unsigned long long)deviceProp.totalGlobalMem);
    printf("GPU Clock arte : %.0f MHz (%.02f " "GHz) \n", deviceProp.clockRate*1e-3f, deviceProp.clockRate*1e-6f);
    printf("Memory Clock rate : %.0f MHz \n", deviceProp.memoryClockRate*1e-3f);
    printf("Memory Bus Width: %d-bit \n", deviceProp.memoryBusWidth);

    if(deviceProp.l2CacheSize)
    {
        printf("L2 Cahceh Size : %d bytes \n", deviceProp.l2CacheSize);
    }

    printf("Max Texture Dimension Size (x, y, z) 1D=(%d), ""2D=(%d,%d), 3D=(%d,%d,%d) \n", deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);

    printf(" Max Layered Texture Size(dim) x layers 1D=(%d) x %d, ""2D=(%d,%d) x %d \n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

    printf("Total amount of constant memory : %lu bytes\n", deviceProp.totalConstMem);

    printf("Total amount of shared memory per block : %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Total numbers o registers available per block : %d \n", deviceProp.regsPerBlock);
    exit(EXIT_SUCCESS);
}

