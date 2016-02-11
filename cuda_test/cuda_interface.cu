#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
extern "C"
cudaError_t cuda_main()
{
	printf("Generating random numbers\n");
	thrust::host_vector<int> h_vec(100);
	thrust::generate(h_vec.begin(), h_vec.end(), rand);
	for(int i=0; i<h_vec.size();i++) printf("%d ",h_vec[i]);
	printf("\n");
	// transfer data to the device
	thrust::device_vector<int> d_vec = h_vec;
	printf("Sorting\n");
	// sort data on the device (805 Mkeys/sec on GeForce GTX 480)
	thrust::sort(d_vec.begin(), d_vec.end());
	// transfer data back to host
	thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
	for(int i=0; i<h_vec.size();i++) printf("%d ",h_vec[i]);
	printf("\nDone\n");
	return cudaGetLastError();
}
