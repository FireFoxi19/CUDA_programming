#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_COLS 32

cudaError_t CudaWork(float* B, float* A, const int width, const int height, const int nreps, const int operation);
void CPUTrans(float* B, float* A, const int width, const int height, const int nreps);

__global__ void DefaultTransKernel(float* B, float* A, const int width, const int height, const int nreps);
__global__ void BankConflictTransKernel(float* B, float* A, const int width, const int height, const int nreps);
__global__ void SharedMemoryTransKernel(float* B, float* A, const int width, const int height, const int nreps);

int main()
{
        int i, width, height, nreps, size, wrong, correct;
        double cpuTime;
        cudaError_t cudaStatus;
        float* A, * ATC, * ATG;

        srand(time(NULL));

        nreps = 10000;



        width = 500;
        height = 100;
        size = width * height;

        A = (float*)malloc(size * sizeof(float));
        ATC = (float*)malloc(size * sizeof(float));
        ATG = (float*)malloc(size * sizeof(float));

        for (i = 0; i < size; i++)
        {
                A[i] = (float)i;
        }

        auto start = std::chrono::high_resolution_clock::now();

        CPUTrans(ATC, A, width, height, nreps);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        cpuTime = (diff.count() * 1000) / nreps;

        correct = 0;
        wrong = 0;

        memset(ATG, 0, size * sizeof(float));
        CudaWork(ATG, A, width, height, nreps, 1);

        for (i = 0; i < size; i++)
        {
                if (ATC[i] != ATG[i])
                {
                        wrong++;
                }
                else
                {
                        correct++;
                }
        }
        correct = 0;
        wrong = 0;

        memset(ATG, 0, size * sizeof(float));
        CudaWork(ATG, A, width, height, nreps, 2);

        for (i = 0; i < size; i++)
        {
                if (ATC[i] != ATG[i])
                {
                        wrong++;
                }
                else
                {
                        correct++;
                }
        }
        correct = 0;
        wrong = 0;

        memset(ATG, 0, size * sizeof(float));
        CudaWork(ATG, A, width, height, nreps, 3);

        for (i = 0; i < size; i++)
        {
                if (ATC[i] != ATG[i])
                {
                        wrong++;
                }
                else
                {
                        correct++;
                }
        }
        correct = 0;
        wrong = 0;

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
                fprintf(stderr, "cudaDeviceReset failed!\n");
                return 1;
        }

        return 0;
}



cudaError_t CudaWork(float* B, float* A, const int width,
        const int height, const int nreps, const int operation)
{
        float elapsed = 0;
        float* dev_A = 0;
        float* dev_B = 0;
        cudaError_t cudaStatus;
        dim3 dim_grid, dim_block;
        double gpuBandwidth;

        int size = width * height;

        dim_block.x = TILE_DIM;
        dim_block.y = BLOCK_ROWS;
        dim_block.z = 1;

        dim_grid.x = (width + TILE_DIM - 1) / TILE_DIM;
        dim_grid.y = (height + TILE_DIM - 1) / TILE_DIM;
        dim_grid.z = 1;

        cudaStatus = cudaSetDevice(0);

        cudaStatus = cudaMalloc((void**)&dev_A, size * sizeof(float));

        cudaStatus = cudaMalloc((void**)&dev_B, size * sizeof(float));

        cudaStatus = cudaMemcpy(dev_A, A, size * sizeof(float),
                cudaMemcpyHostToDevice);

        cudaMemset(dev_B, 0, size * sizeof(float));
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        switch (operation)
        {
        case(1):
        {
                cudaEventRecord(start);
                DefaultTransKernel << <dim_grid, dim_block >> > (dev_B, dev_A, width, height, nreps);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&elapsed, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                //elapsed /= nreps;
                printf("Default: %fms\n", elapsed);

                break;
        }

        case(2):
        {
                cudaEventRecord(start);
                BankConflictTransKernel << <dim_grid, dim_block >> > (dev_B, dev_A, width, height, nreps);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&elapsed, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                //elapsed /= nreps;

                printf("Bank Conflict: %fms\n",elapsed);
                break;
        }
        case(3):
        {
                /**/
                cudaEventRecord(start);
                SharedMemoryTransKernel << <dim_grid, dim_block >> > (dev_B, dev_A, width, height, nreps);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                cudaEventElapsedTime(&elapsed, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);

                //elapsed /= nreps;

                printf("Shared memory: %fms\n",elapsed);

        }
        }

        cudaStatus = cudaGetLastError();


        cudaStatus = cudaDeviceSynchronize();

        cudaStatus = cudaMemcpy(B, dev_B, size * sizeof(float),
                cudaMemcpyDeviceToHost);


        return cudaStatus;
}

void CPUTrans(float* B, float* A, const int width, const int
        height, const int nreps)
{
        int i, j, r;

#pragma unroll
        for (r = 0; r < nreps; r++)
#pragma unroll
                for (i = 0; i < height; i++)
#pragma unroll
                        for (j = 0; j < width; j++)
                                B[j * height + i] = A[i * width + j];
}

__global__ void DefaultTransKernel(float* B, float* A, const int width, const int height, const int nreps)
{
        int i, r;
        int col = blockIdx.x * TILE_DIM + threadIdx.x;
        int row = blockIdx.y * TILE_DIM + threadIdx.y;
        int index_in = col + width * row;
        int index_out = row + height * col;

#pragma unroll
        for (r = 0; r < nreps; r++)
#pragma unroll
                for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
                        if ((row + i < height) && (col < width))
                                B[index_out + i] = A[index_in + i * width];
}

__global__ void BankConflictTransKernel(float* B, float* A, const int width, const int height, const int nreps)
{
        __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // кэш в разделяемой памяти - прибавили один стобец(увеличили кэш)
        int ciIndex = blockIdx.x * TILE_DIM + threadIdx.x;
        int riIndex = blockIdx.y * TILE_DIM + threadIdx.y;
        int coIndex = blockIdx.y * TILE_DIM + threadIdx.x;
        int roIndex = blockIdx.x * TILE_DIM + threadIdx.y;
        int index_in = ciIndex + (riIndex)*width;
        int index_out = coIndex + (roIndex)*height; int r, i;
#pragma unroll
        for (r = 0; r < nreps; r++)  // транспонирование
        {
#pragma unroll
                for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
                        if ((ciIndex < width) && (riIndex + i < height))
                                tile[threadIdx.y + i][threadIdx.x] = A[index_in + i * width];
                __syncthreads();

#pragma unroll
                for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
                        if ((coIndex < height) && (roIndex + i < width))
                                B[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
                __syncthreads();
        }
}

__global__ void SharedMemoryTransKernel(float* B, float* A, const int width, const int height, const int nreps)
{
        __shared__ float tile[TILE_DIM][TILE_DIM]; //  кэш в разделяемой памяти
        int ciIndex = blockIdx.x * TILE_DIM + threadIdx.x;
        int riIndex = blockIdx.y * TILE_DIM + threadIdx.y;
        int coIndex = blockIdx.y * TILE_DIM + threadIdx.x;
        int roIndex = blockIdx.x * TILE_DIM + threadIdx.y;
        int index_in = ciIndex + (riIndex)*width;
        int index_out = coIndex + (roIndex)*height; int r, i;
#pragma unroll
        for (r = 0; r < nreps; r++) // транспонирование
        {
#pragma unroll
                for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
                        if ((ciIndex < width) && (riIndex + i < height))
                                tile[threadIdx.y + i][threadIdx.x] = A[index_in + i * width];
                __syncthreads();

#pragma unroll
                for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
                        if ((coIndex < height) && (roIndex + i < width))
                                B[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
                __syncthreads();
        }
}
