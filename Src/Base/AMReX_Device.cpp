
#include <AMReX_Device.H>

bool amrex::Device::in_device_launch_region = false;

#ifdef CUDA
int amrex::Device::cuda_device_id = 0;

cudaStream_t amrex::Device::cuda_streams[max_cuda_streams];

void
amrex::Device::c_threads_and_blocks(const int* lo, const int* hi, dim3& numBlocks, dim3& numThreads) {

    int tile_size[BL_SPACEDIM];

    for (int i = 0; i < BL_SPACEDIM; ++i)
	tile_size[i] = hi[i] - lo[i] + 1;

#if (BL_SPACEDIM == 1)

	numThreads.x = 256;
	numThreads.y = 1;
	numThreads.z = 1;

	numBlocks.x = (tile_size[0] + numThreads.x - 1) / numThreads.x;
	numBlocks.y = 1;
	numBlocks.z = 1;

#elif (BL_SPACEDIM == 2)

	numThreads.x = 16;
	numThreads.y = 16;
	numThreads.z = 1;

	numBlocks.x = (tile_size[0] + numThreads.x - 1) / numThreads.x;
	numBlocks.y = (tile_size[1] + numThreads.y - 1) / numThreads.y;
	numBlocks.z = 1;

#else

	numThreads.x = 8;
	numThreads.y = 8;
	numThreads.z = 8;

	numBlocks.x = (tile_size[0] + numThreads.x - 1) / numThreads.x;
	numBlocks.y = (tile_size[1] + numThreads.y - 1) / numThreads.y;
	numBlocks.z = (tile_size[2] + numThreads.z - 1) / numThreads.z;

#endif

}

void
amrex::Device::initialize_cuda_c () {

    for (int i = 0; i < max_cuda_streams; ++i)
	cudaStreamCreate(&cuda_streams[i]);

}

cudaStream_t
amrex::Device::stream_from_index(int idx) {

    if (idx < 0)
	return 0;
    else
	return cuda_streams[idx % 100];

}

#endif
	   
