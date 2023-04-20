#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

cudaError_t weirdCopyWithCuda2(int* a, int* b, int rows, int cols);
cudaError_t weirdCopyWithCuda(int* a, int* b, int rows, int cols);
cudaError_t fillWithCuda(int *a, int size);

__global__ void fillKernel(int *a, int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a[i] = i;
}

__global__ void weirdCopyKernel(int *a, int *b, int rows, int cols) // ���������������� 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; // 1D ���� 1D �����
    int row = i / cols; // ����������� ������� row � col ��� ���������� �������� � ����� (� ������ ����� ������ ���������� � 0 �� (���-�� ������� � �����))
    int col = i % cols;
    b[col * rows + row] = a[i]; // ����������� �������� ������� b ������� ������� a 
}


__global__ void weirdCopyKernel2(int *a, int *b, int rows, int cols) // �����������
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int row = i / cols;
    int col = i % cols;
    b[col + row * cols] = a[i];
}

int main()
{
    const int rows = 3;
    const int cols = 4;
    int arraySize = rows * cols;
    int a[rows * cols];
    int b[rows * cols];

    cudaError_t cudaStatus = fillWithCuda(a, arraySize); // ���������� ������� a �� GPU 

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fillWithCuda failed!");
        return 1;
    }

    printf("A: ");
    for(int i = 0; i < arraySize; i++){
        printf("%d ", a[i]);
    }
    printf("\n");

    // Add vectors in parallel.
    cudaStatus = weirdCopyWithCuda(a, b, rows, cols); // ���������������� ������� a � b(�����������) �� GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "weirdCopyWithCuda failed!");
        return 1;
    }

    printf("B: ");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");

    cudaStatus = weirdCopyWithCuda2(a, b, rows, cols); // ������ ����������� ������� � � b �� GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "weirdCopyWithCuda failed!");
        return 1;
    }

    printf("A: ");
    for(int i = 0; i < arraySize; i++){
        printf("%d ", a[i]);
    }
    printf("\n");


    printf("B: ");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t fillWithCuda(int *a, int size) // ���������� ������� �� GPU
{
    cudaError_t cudaStatus;
    int *dev_a = NULL; // ������� ���������(������) ���_�

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); // �������� ������ �� GPU ��� ������ ���_�

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); // �������� ������ � � ���_�
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    fillKernel<<<1, size>>>(dev_a, size);// ������� ���������� ������� ���_�


    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "fillKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost); // �������� ������ ���_� � �
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);

    return cudaStatus;
}

cudaError_t weirdCopyWithCuda2(int* a, int* b, int rows, int cols) // ������� ����������� �������
{
    int *dev_a = 0;
    int* dev_b = 0;
    int size = rows * cols;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    weirdCopyKernel2<<<rows, cols>>>(dev_a, dev_b, rows, cols);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// ��������������� ������� ��� ������������� CUDA ��� ������������� ���������� ��������.
cudaError_t weirdCopyWithCuda(int* a, int* b, int rows, int cols) // ���������������� ������� (����������� � � b)
{
    int *dev_a = 0;
    int* dev_b = 0;
    int size = rows * cols;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); // ��������� ������ �� GPU ��� ������ ���_�
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int)); // ��������� ������ �� GPU ��� ������ ���_b
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); // ����������� ������ �� GPU � ���_�
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // ������ ���� �� ����������� ���������� � ����� ������� ��� ������� ��������.
    weirdCopyKernel<<<rows, cols>>>(dev_a, dev_b, rows, cols); // ������� ���������������� row(���-�� �������) cols(���-�� ������)

    // ��������, ��� �� ������ ��� ������� ����
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    //cudaDeviceSynchronize ������� ���������� ������ ���� � ���������� ��� ������, ��������� �� ����� �������.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost); // ����������� ���������� ���������������� �� ���_b � b 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
