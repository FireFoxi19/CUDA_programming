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

__global__ void weirdCopyKernel(int *a, int *b, int rows, int cols) // “ранспонирование 
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; // 1D блок 1D поток
    int row = i / cols; // высчитываем индексы row и col дл€ нахождение элемента в блоке (в каждом блоке потоки нумируютс€ с 0 до (кол-во потоков в блоке))
    int col = i % cols;
    b[col * rows + row] = a[i]; // присваиваем элементу матрицы b элемент матрицы a 
}


__global__ void weirdCopyKernel2(int *a, int *b, int rows, int cols) //  опирование
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

    cudaError_t cudaStatus = fillWithCuda(a, arraySize); // «аполнение массива a на GPU 

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
    cudaStatus = weirdCopyWithCuda(a, b, rows, cols); // транспонирование матрицы a в b(копирование) на GPU
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "weirdCopyWithCuda failed!");
        return 1;
    }

    printf("B: ");
    for (int i = 0; i < arraySize; i++) {
        printf("%d ", b[i]);
    }
    printf("\n");

    cudaStatus = weirdCopyWithCuda2(a, b, rows, cols); // просто копирование матрицы а в b на GPU
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

cudaError_t fillWithCuda(int *a, int size) // «аполнение массива на GPU
{
    cudaError_t cudaStatus;
    int *dev_a = NULL; // —оздает указатель(массив) дев_а

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); // выдел€ет пам€ть на GPU под массив дев_а

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); // копирует массив а в дев_а
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    fillKernel<<<1, size>>>(dev_a, size);// ‘ункци€ заполнени€ массива дев_а


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

    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost); // копирует массив дев_а в а
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);

    return cudaStatus;
}

cudaError_t weirdCopyWithCuda2(int* a, int* b, int rows, int cols) // простое копирование массива
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

// ¬спомогательна€ функци€ дл€ использовани€ CUDA дл€ параллельного добавлени€ векторов.
cudaError_t weirdCopyWithCuda(int* a, int* b, int rows, int cols) // транспонирование матрицы (копирование а в b)
{
    int *dev_a = 0;
    int* dev_b = 0;
    int size = rows * cols;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int)); // выделение пам€ти на GPU под массив дев_а
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int)); // выделение пам€ти на GPU под массив дев_b
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice); // копирование пам€ти на GPU в дев_а
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // «апуск €дра на графическом процессоре с одним потоком дл€ каждого элемента.
    weirdCopyKernel<<<rows, cols>>>(dev_a, dev_b, rows, cols); // функци€ транспонировани€ row(кол-во потоков) cols(кол-во блоков)

    // ѕроверка, нет ли ошибок при запуске €дра
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    //cudaDeviceSynchronize ожидает завершени€ работы €дра и возвращает все ошибки, возникшие во врем€ запуска.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(int), cudaMemcpyDeviceToHost); // копирование результата транспонировани€ из дев_b в b 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
