#include <iostream>
#include <cuda.h>
#include <builtin_types.h>
#include <stdio.h>
using namespace std;
// the next 'include' is for CUDA error checks
#include <cuda_runtime.h>
extern "C"
cudaError_t cuda_main();

int main(int argc, char *argv[])
{
	int deviceCount = 0;
	int cudaDevice = 0;
	char cudaDeviceName [100];

	cuInit(0);
	cuDeviceGetCount(&deviceCount);
	cuDeviceGet(&cudaDevice, 0);
	cuDeviceGetName(cudaDeviceName, 100, cudaDevice);
	cout << "Number of devices: " << deviceCount << endl;
	cout << "Device name:" << cudaDeviceName << endl;
	// run your cuda application
	cudaError_t cuerr = cuda_main();
	// check for errors is always a good practice!
	if (cuerr != cudaSuccess) cout << "CUDA Error: " << cudaGetErrorString( cuerr ) << endl;
	printf("Done\n");
	return 0;
}
