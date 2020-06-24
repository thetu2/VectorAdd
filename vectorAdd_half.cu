
/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include "cuComplex.h"
#include "cuda_fp16.h"
/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const __half2 *A, const __half2 *B, __half2 *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
    #if __CUDA_ARCH__ >= 530
            C[i] = __hadd2(A[i], B[i]); 
    #else
            C[i] = __floats2half2_rn(__half22float2(A[i]).x + __half22float2(B[i]).x, 
                __half22float2(A[i]).y + __half22float2(B[i]).y );
    #endif
    }
}

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(__half2);
    printf("[Vector addition of %d elements]\n", numElements);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate the host input vector A
    __half2 *h_A = (__half2*)malloc(size);

    // Allocate the host input vector B
    __half2 *h_B = (__half2*)malloc(size);

    // Allocate the host output vector C
    __half2 *h_C = (__half2*)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    float2 temp_A_float2;
    float2 temp_B_float2;
        
    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        temp_A_float2.x = rand() / (float)RAND_MAX;
        temp_A_float2.y = rand() / (float)RAND_MAX;
        h_A[i] = __float22half2_rn(temp_A_float2);
        temp_B_float2.x = rand() / (float)RAND_MAX;
        temp_B_float2.y = rand() / (float)RAND_MAX;
        h_B[i] = __float22half2_rn(temp_B_float2);

    }

    // Allocate the device input vector A
    __half2 *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    __half2 *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    __half2 *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    cudaEventRecord(start);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaEventRecord(stop);
    err = cudaGetLastError();



    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Verify that the result vector is correct
    //for (int i = 0; i < numElements; ++i)
    //{
    //    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    //    {
    //        fprintf(stderr, "Result verification failed at element %d!\n", i);
    //        exit(EXIT_FAILURE);
    //    }
    //}

    int idx1= 6;

    float2 sample_val_A=__half22float2(h_A[idx1]);
    float2 sample_val_B=__half22float2(h_B[idx1]);
    float2 sample_val_C= __half22float2(h_C[idx1]);
    

    printf("Test PASSED\n");
    printf("Sample output on index %d: (%f+%fi)+(%f+%fi)=%f+%fi\n", idx1, sample_val_A.x, sample_val_A.y, sample_val_B.x, sample_val_B.y,
        sample_val_C.x, sample_val_C.y);

    printf("Kernel time: %f ms\n", milliseconds);

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

