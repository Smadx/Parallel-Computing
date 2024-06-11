#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstddef>
#include <cstdlib>

using namespace std;

__global__ void spmm(int *dense, int *sparse, int *result, size_t pitch, int M, int N, int P, int K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < P)
    {
        int sum = 0;
        int *sparse_col = (int *)((char *)sparse + col * pitch);
        for (int i = 0; i < sparse_col[0]; i++)
        {
            sum += dense[row * N + sparse_col[i * 2 + 1]] * sparse_col[i * 2 + 2];
        }
        result[row * P + col] = sum;
    }
}

int main()
{
    int M, N, P, K;
    cin >> M >> N >> P >> K;
    int *dense = (int *)malloc(M * N * sizeof(int));
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cin >> dense[i * N + j];
        }
    }

    // CSC
    int *sparse = (int *)malloc(P * (2 * K + 1) * sizeof(int));
    for (int i = 0; i < P; i++)
    {
        sparse[i * (2 * K + 1)] = 0;
    }
    for (int i = 0; i < K; i++)
    {
        int row, col, val;
        cin >> row >> col >> val;
        sparse[col * (2 * K + 1)]++;
        sparse[col * (2 * K + 1) + sparse[col * (2 * K + 1)] * 2 - 1] = row;
        sparse[col * (2 * K + 1) + sparse[col * (2 * K + 1)] * 2] = val;
    }

    int *result = (int *)malloc(M * P * sizeof(int));

    int *d_dense, *d_sparse, *d_result;
    cudaMalloc(&d_dense, M * N * sizeof(int));
    size_t pitch;
    cudaMallocPitch((void **)&d_sparse, &pitch, (2 * K + 1) * sizeof(int), P);
    cudaMalloc(&d_result, M * P * sizeof(int));

    cudaMemcpy(d_dense, dense, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_sparse, pitch, sparse, (2 * K + 1) * sizeof(int), (2 * K + 1) * sizeof(int), P, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (P + threadsPerBlock.y - 1) / threadsPerBlock.y);

    spmm<<<numBlocks, threadsPerBlock>>>(d_dense, d_sparse, d_result, pitch, M, N, P, K);

    cudaMemcpy(result, d_result, M * P * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_dense);
    cudaFree(d_sparse);
    cudaFree(d_result);

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < P; j++)
        {
            cout << result[i * P + j] << " ";
        }
        cout << endl;
    }
    return 0;
}