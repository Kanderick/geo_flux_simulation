/*
Given a grid of particular size and the physical property value of every element in the grid,
this code calculates the flux of every adjacent surfaces of the grid.
Both a sequential and a parallel implementation are provided.
*/

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define SURFACE_COUNT 6
#define GRID_DIMENSION 10
#define GRID_SIZE GRID_DIMENSION*GRID_DIMENSION*GRID_DIMENSION
#define AREA 1
#define BLOCK_SIZE 1024
#define DEBUG_PRINT 0


/*
initializes the grid input parameter
In this case, all parameters in z and y dimension are the same,
parameters in x dimension are the same as their x dimension index.
*/
void grid_init(float* property) {
	for (int z = 0; z < GRID_DIMENSION; z++) {
		for (int y = 0; y < GRID_DIMENSION; y++) {
			for (int x = 0; x < GRID_DIMENSION; x++) {
				property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] = 2 * (float)y * (float)z + (float)x + 2 * (float)y;
			}
		}
	}
}

/*
sequentially calculates the flux given the input property array,
and populates the flux array with output.
*/
void calculateFluxCPU(float* property, float* flux) {
	//loop for six surfaces
	for (int i = 0; i < SURFACE_COUNT; i++) {
		//loop for all elements through that surface
		for (int z = 0; z < GRID_DIMENSION; z++) {
			for (int y = 0; y < GRID_DIMENSION; y++) {
				for (int x = 0; x < GRID_DIMENSION; x++) {
					switch (i) {
					case 0:
						flux[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] = \
							(z == 0) ? 0.0 : (property[(z - 1) * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] - \
								property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]) * AREA;
						break;
					case 1:
						flux[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] = \
							(z == GRID_DIMENSION - 1) ? 0.0 : (property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] - \
								property[(z+1) * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]) * AREA;
						break;
					case 2:
						flux[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] = \
							(y == 0) ? 0.0 : (property[z * GRID_DIMENSION * GRID_DIMENSION + (y-1) * GRID_DIMENSION + x] - \
								property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]) * AREA;
						break;
					case 3:
						flux[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] = \
							(y == GRID_DIMENSION - 1) ? 0.0 : (property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] - \
								property[z * GRID_DIMENSION * GRID_DIMENSION + (y+1) * GRID_DIMENSION + x]) * AREA;
						break;
					case 4:
						flux[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] = \
							(x == 0) ? 0.0 : (property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + (x-1)] - \
								property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]) * AREA;
						break;
					case 5:
						flux[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] = \
							(x == GRID_DIMENSION - 1) ? 0.0 : (property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] - \
								property[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + (x+1)]) * AREA;
						break;
					}
				}
			}
		}
	}
}

/*
parallelly calculates the flux given the input property array,
and populates the flux array with output.
*/
__global__ void calculateFluxGPU (float* property, float* flux) {
	/*shared memory version of properety array*/
	__shared__ float prop[GRID_DIMENSION][GRID_DIMENSION][GRID_DIMENSION];

	/*each thread collablratively loads all data into shared memory*/
	prop[threadIdx.z][threadIdx.y][threadIdx.x] = property[threadIdx.z * GRID_DIMENSION * GRID_DIMENSION + threadIdx.y * GRID_DIMENSION + threadIdx.x];
	__syncthreads();

	switch (blockIdx.x) {
	case 0:
		flux[blockIdx.x * GRID_SIZE + threadIdx.z * GRID_DIMENSION * GRID_DIMENSION + threadIdx.y * GRID_DIMENSION + threadIdx.x] = \
			(threadIdx.z == 0) ? 0.0 : (prop[threadIdx.z - 1][threadIdx.y][threadIdx.x] - prop[threadIdx.z][threadIdx.y][threadIdx.x]) * AREA;
		break;
	case 1:
		flux[blockIdx.x * GRID_SIZE + threadIdx.z * GRID_DIMENSION * GRID_DIMENSION + threadIdx.y * GRID_DIMENSION + threadIdx.x] = \
			(threadIdx.z == GRID_DIMENSION - 1) ? 0.0 : (prop[threadIdx.z][threadIdx.y][threadIdx.x] - prop[threadIdx.z + 1][threadIdx.y][threadIdx.x]) * AREA;
		break;
	case 2:
		flux[blockIdx.x * GRID_SIZE + threadIdx.z * GRID_DIMENSION * GRID_DIMENSION + threadIdx.y * GRID_DIMENSION + threadIdx.x] = \
			(threadIdx.y == 0) ? 0.0 : (prop[threadIdx.z][threadIdx.y - 1][threadIdx.x] - prop[threadIdx.z][threadIdx.y][threadIdx.x]) * AREA;
		break;
	case 3:
		flux[blockIdx.x * GRID_SIZE + threadIdx.z * GRID_DIMENSION * GRID_DIMENSION + threadIdx.y * GRID_DIMENSION + threadIdx.x] = \
			(threadIdx.y == GRID_DIMENSION - 1) ? 0.0 : (prop[threadIdx.z][threadIdx.y][threadIdx.x] - prop[threadIdx.z][threadIdx.y + 1][threadIdx.x]) * AREA;
		break;
	case 4:
		flux[blockIdx.x * GRID_SIZE + threadIdx.z * GRID_DIMENSION * GRID_DIMENSION + threadIdx.y * GRID_DIMENSION + threadIdx.x] = \
			(threadIdx.x == 0) ? 0.0 : (prop[threadIdx.z][threadIdx.y][threadIdx.x - 1] - prop[threadIdx.z][threadIdx.y][threadIdx.x]) * AREA;
		break;
	case 5:
		flux[blockIdx.x * GRID_SIZE + threadIdx.z * GRID_DIMENSION * GRID_DIMENSION + threadIdx.y * GRID_DIMENSION + threadIdx.x] = \
			(threadIdx.x == GRID_DIMENSION - 1) ? 0.0 : (prop[threadIdx.z][threadIdx.y][threadIdx.x] - prop[threadIdx.z][threadIdx.y][threadIdx.x + 1]) * AREA;
		break;
	}

}

/*function to check the correctness of the GPU output*/
int correctness_check(float* flux1, float* flux2) {
	int correct = 1;

	for (int i = 0; i < SURFACE_COUNT; i++) {
		//loop for all elements through that surface
		for (int z = 0; z < GRID_DIMENSION; z++) {
			for (int y = 0; y < GRID_DIMENSION; y++) {
				for (int x = 0; x < GRID_DIMENSION; x++) {
					if (flux1[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x] != \
						flux2[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]) {
						correct = 0;
						break;
					}
				}
				if (!correct) break;
			}
			if (!correct) break;
		}
		if (!correct) break;
	}
	return correct;
}


int main() {
	/*parameters*/

	/*property array is of size GRID_DIMENSION cubed, ordered in the hierarchy of <z, y, x>*/
	float* hostProperty;
	/*flux array is of size 6 * GRID_DIMENSION cubed, ordered sequentially with six contiguous
	regions of size GRID_DIMENTION cubed, the six regions respectively records the flux in
	z+, z-, y+, y-, x+ and x- direction*/
	float* hostFluxCPU;
	float* hostFluxGPU;
	float* deviceProperty;
	float* deviceFlux;

	int i;
	int deviceCount;
	cudaDeviceProp prop;

	cudaGetDeviceCount(&deviceCount);
	printf("Device Count: %d\n\n", deviceCount);

	for (i = 0; i < deviceCount; i++) {
		printf("Device Number: %d\n", i);

		cudaGetDeviceProperties(&prop, i);

		printf("Device Name: %s\n", prop.name);
		printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock Rate(kHz): %d\n", prop.clockRate);
		printf("Memory Clock Rate(kHz): %d\n", prop.memoryClockRate);
		printf("Memory Bus Width(b): %d\n", prop.memoryBusWidth);
		printf("Total global Memory(B): %lu\n", prop.totalGlobalMem);
		printf("Total Constant Memory(B): %lu\n", prop.totalConstMem);
		printf("L2 Cache Size(B): %d\n", prop.l2CacheSize);
		printf("Multiprocessor Count: %d\n", prop.multiProcessorCount);
		printf("Shared Memory Per Multiprocessor(B): %lu\n", prop.sharedMemPerMultiprocessor);
		printf("Shared Memory Per Block(B): %lu\n", prop.sharedMemPerBlock);
		printf("Regs Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
		printf("Regs Per Block: %d\n", prop.regsPerBlock);
		printf("Warp Size: %d\n", prop.warpSize);
		printf("Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
		printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
		printf("Max Grid Dimension: <%d, %d, %d>\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Max Block Dimension: <%d, %d, %d>\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("\n");
	}

	/*allocates pinned host memory for parameters*/
	cudaMallocHost(&hostProperty, GRID_SIZE * sizeof(float));
	cudaMallocHost(&hostFluxCPU, SURFACE_COUNT * GRID_SIZE * sizeof(float));
	cudaMallocHost(&hostFluxGPU, SURFACE_COUNT * GRID_SIZE * sizeof(float));
	cudaMalloc(&deviceProperty, GRID_SIZE * sizeof(float));
	cudaMalloc(&deviceFlux, SURFACE_COUNT * GRID_SIZE * sizeof(float));

	/*initializes the host input parameters*/
	grid_init(hostProperty);

	/*prints out the input parameters*/
	if (DEBUG_PRINT) {
		for (int z = 0; z < GRID_DIMENSION; z++) {
			for (int y = 0; y < GRID_DIMENSION; y++) {
				for (int x = 0; x < GRID_DIMENSION; x++) {
					printf("Property<%d,%d,%d> = %f\n", z, y, x, hostProperty[z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]);
				}
			}
		}
		printf("\n");
	}

	/*timeing the CPU execution*/
	LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
	LARGE_INTEGER Frequency;

	QueryPerformanceFrequency(&Frequency);
	QueryPerformanceCounter(&StartingTime);

	/*sequentially calculates flux and populate the hostFluxCPU array*/
	calculateFluxCPU(hostProperty, hostFluxCPU);

	QueryPerformanceCounter(&EndingTime);
	ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;
	ElapsedMicroseconds.QuadPart *= 1000000;
	ElapsedMicroseconds.QuadPart /= Frequency.QuadPart;
	double elapsedMillisecond = ElapsedMicroseconds.QuadPart / 1000.0;
	printf("calculateFluxCPU() Execution Time(ms): %lf\n", elapsedMillisecond);

	/*prints out the CPU output parameters*/
	if (DEBUG_PRINT) {
		for (int i = 0; i < SURFACE_COUNT; i++) {
			//loop for all elements through that surface
			for (int z = 0; z < GRID_DIMENSION; z++) {
				for (int y = 0; y < GRID_DIMENSION; y++) {
					for (int x = 0; x < GRID_DIMENSION; x++) {
						printf("Flux_CPU<%d,%d,%d,%d> = %f\n", i, z, y, x, hostFluxCPU[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]);
					}
				}
			}
		}
		printf("\n");
	}

	/*Memory copy to cuda device*/
	cudaMemcpy(deviceProperty, hostProperty, GRID_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	/*lauch parameters, need modification if setting is changed*/
	dim3 DimBlock(GRID_DIMENSION, GRID_DIMENSION, GRID_DIMENSION);
	dim3 DimGrid(SURFACE_COUNT, 1, 1);

	/*launch the kernel to calculate the flux, only applies to a grid of dimenision less or equal to 10 in order to fit into one block,
	need modification if processing grid of larger dimensions*/
	/*for kernel timing*/
	cudaEvent_t executeStartGPU, executeEndGPU;
	cudaEventCreate(&executeStartGPU);
	cudaEventCreate(&executeEndGPU);
	cudaEventRecord(executeStartGPU);

	calculateFluxGPU << <DimGrid, DimBlock >> > (deviceProperty, deviceFlux);

	cudaEventRecord(executeEndGPU);

	/*copy device output back to host*/
	cudaMemcpy(hostFluxGPU, deviceFlux, SURFACE_COUNT * GRID_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	/*prints out kernel call execution time*/
	cudaEventSynchronize(executeEndGPU);
	float t2 = 0;
	cudaEventElapsedTime(&t2, executeStartGPU, executeEndGPU);
	cudaEventDestroy(executeStartGPU);
	cudaEventDestroy(executeEndGPU);
	printf("calculateFluxGPU() Execution Time(ms): %f\n", t2);
	printf("\n");

	/*prints out the GPU output parameters*/
	if (DEBUG_PRINT) {
		for (int i = 0; i < SURFACE_COUNT; i++) {
			//loop for all elements through that surface
			for (int z = 0; z < GRID_DIMENSION; z++) {
				for (int y = 0; y < GRID_DIMENSION; y++) {
					for (int x = 0; x < GRID_DIMENSION; x++) {
						printf("Flux_GPU<%d,%d,%d,%d> = %f\n", i, z, y, x, hostFluxGPU[i * GRID_SIZE + z * GRID_DIMENSION * GRID_DIMENSION + y * GRID_DIMENSION + x]);
					}
				}
			}
		}
		printf("\n");
	}

	/*compare the correctness of gpu output with that of cpu*/
	if (correctness_check(hostFluxCPU, hostFluxGPU)) {
		printf("[SUCCESS] Output of GPU is consistent with that of CPU\n");
	}
	else {
		printf("[Fail] Output of GPU is inconsistent with that of CPU\n");
	}
	printf("\n");

	/*free parameters*/
	cudaFree(deviceProperty);
	cudaFree(deviceFlux);
	cudaFreeHost(hostProperty);
	cudaFreeHost(hostFluxCPU);
	cudaFreeHost(hostFluxGPU);

	return 0;
}