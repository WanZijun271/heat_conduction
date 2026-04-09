#include "kernels.cuh"
#include "config.h"
#include "constants.h"

using namespace std;

__global__ void JacobiIterateKernel(scalar *temp, scalar* temp0, scalar *coef, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idP = i + j * nx + k * nx * ny;
        scalar tP = temp0[idP];
        // tE
        scalar tE = 0.0;
        if ( i != nx - 1 ) {
            int idE = (i+1) + j * nx + k * nx * ny;
            tE = temp0[idE];
        }
        // tW
        scalar tW = 0.0;
        if ( i != 0 ) {
            int idW = (i-1) + j * nx + k * nx * ny;
            tW = temp0[idW];
        }
        // tN
        scalar tN = 0.0;
        if ( j != ny - 1 ) {
            int idN = i + (j+1) * nx + k * nx * ny;
            tN = temp0[idN];
        }
        // tN
        scalar tS = 0.0;
        if ( j != 0 ) {
            int idS = i + (j-1) * nx + k * nx * ny;
            tS = temp0[idS];
        }
        // tT
        scalar tT = 0.0;
        if ( k != nz - 1 ) {
            int idT = i + j * nx + (k+1) * nx * ny;
            tT = temp0[idT];
        }
        // tB
        scalar tB = 0.0;
        if ( k != 0 ) {
            int idB = i + j * nx + (k-1) * nx * ny;
            tB = temp0[idB];
        }
        
        scalar tNew = tP;
        if (dim == 2) {
            int id = i * 6 + j * nx * 6;
            tNew = coef[id+id_b]
                - coef[id+id_aE] * tE
                - coef[id+id_aW] * tW
                - coef[id+id_aN] * tN
                - coef[id+id_aS] * tS;
            tNew /= coef[id+id_aP];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            tNew = coef[id+id_b]
                - coef[id+id_aE] * tE
                - coef[id+id_aW] * tW
                - coef[id+id_aN] * tN
                - coef[id+id_aS] * tS
                - coef[id+id_aT] * tT
                - coef[id+id_aB] * tB;
            tNew /= coef[id+id_aP];
        }

        scalar dT = relax * (tNew - tP);

        temp[idP] = tP + dT;

        sharedNorm[threadId] = dT*dT;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                sharedNorm[threadId] += sharedNorm[threadId + s];
            }
            __syncthreads();
        }

        if (threadId == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void JacobiIterate(vector<scalar>& temp, const vector<scalar>& coef) {

    size_t tempSize = temp.size() * sizeof(scalar);
    size_t coefSize = coef.size() * sizeof(scalar);

    scalar *devTemp, *devTemp0, *devCoef, *devNorm;
    cudaMalloc(&devTemp, tempSize);
    cudaMalloc(&devTemp0, tempSize);
    cudaMalloc(&devCoef, coefSize);
    cudaMalloc(&devNorm, sizeof(scalar));

    cudaMemcpy(devTemp, temp.data(), tempSize, cudaMemcpyHostToDevice);
    cudaMemset(devTemp0, 0, tempSize);
    cudaMemcpy(devCoef, coef.data(), coefSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        threadsPerBlock.z = 1;

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 8;

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < niter; ++it) {
        
        scalar *tmp = devTemp;
        devTemp = devTemp0;
        devTemp0 = tmp;

        scalar norm = 0.0;
        cudaMemcpy(devNorm, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        JacobiIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(devTemp, devTemp0, devCoef, devNorm);
        
        cudaMemcpy(&norm, devNorm, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaMemcpy(temp.data(), devTemp, tempSize, cudaMemcpyDeviceToHost);

    cudaFree(devTemp);
    cudaFree(devTemp0);
    cudaFree(devCoef);
    cudaFree(devNorm);
}

__global__ void GaussSeidelIterateKernel(scalar *temp, scalar *coef, scalar *norm) {

    extern __shared__ scalar sharedNorm[];

    int threadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idP = i + j * nx + k * nx * ny;
        scalar tP = temp[idP];
        // tE
        scalar tE = 0.0;
        if ( i != nx - 1 ) {
            int idE = (i+1) + j * nx + k * nx * ny;
            tE = temp[idE];
        }
        // tW
        scalar tW = 0.0;
        if ( i != 0 ) {
            int idW = (i-1) + j * nx + k * nx * ny;
            tW = temp[idW];
        }
        // tN
        scalar tN = 0.0;
        if ( j != ny - 1 ) {
            int idN = i + (j+1) * nx + k * nx * ny;
            tN = temp[idN];
        }
        // tN
        scalar tS = 0.0;
        if ( j != 0 ) {
            int idS = i + (j-1) * nx + k * nx * ny;
            tS = temp[idS];
        }
        // tT
        scalar tT = 0.0;
        if ( k != nz - 1 ) {
            int idT = i + j * nx + (k+1) * nx * ny;
            tT = temp[idT];
        }
        // tB
        scalar tB = 0.0;
        if ( k != 0 ) {
            int idB = i + j * nx + (k-1) * nx * ny;
            tB = temp[idB];
        }
        
        scalar tNew = tP;
        if (dim == 2) {
            int id = i * 6 + j * nx * 6;
            tNew = coef[id+id_b]
                - coef[id+id_aE] * tE
                - coef[id+id_aW] * tW
                - coef[id+id_aN] * tN
                - coef[id+id_aS] * tS;
            tNew /= coef[id+id_aP];
        } else if (dim == 3) {
            int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
            tNew = coef[id+id_b]
                - coef[id+id_aE] * tE
                - coef[id+id_aW] * tW
                - coef[id+id_aN] * tN
                - coef[id+id_aS] * tS
                - coef[id+id_aT] * tT
                - coef[id+id_aB] * tB;
            tNew /= coef[id+id_aP];
        }

        scalar dT = relax * (tNew - tP);

        temp[idP] = tP + dT;

        sharedNorm[threadId] = dT*dT;
        __syncthreads();

        int stride = blockDim.x * blockDim.y * blockDim.z;
        for (int s = stride / 2; s > 0; s >>= 1) {
            if (threadId < s) {
                sharedNorm[threadId] += sharedNorm[threadId + s];
            }
            __syncthreads();
        }

        if (threadId == 0) {
            atomicAdd(norm, sharedNorm[0]);
        }
    }
}

void GaussSeidelIterate(vector<scalar>& temp, const vector<scalar>& coef) {

    size_t tempSize = temp.size() * sizeof(scalar);
    size_t coefSize = coef.size() * sizeof(scalar);

    scalar *devTemp, *devCoef, *devNorm;
    cudaMalloc(&devTemp, tempSize);
    cudaMalloc(&devCoef, coefSize);
    cudaMalloc(&devNorm, sizeof(scalar));

    cudaMemcpy(devTemp, temp.data(), tempSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devCoef, coef.data(), coefSize, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock;
    dim3 numBlocks;
    if (dim == 2) {
        threadsPerBlock.x = 32;
        threadsPerBlock.y = 32;
        threadsPerBlock.z = 1;

        numBlocks.x = (nx + 31) / 32;
        numBlocks.y = (ny + 31) / 32;
        numBlocks.z = 1;
    } else if (dim == 3) {
        threadsPerBlock.x = 16;
        threadsPerBlock.y = 8;
        threadsPerBlock.z = 8;

        numBlocks.x = (nx + 15) / 16;
        numBlocks.y = (ny + 7) / 8;
        numBlocks.z = (nz + 7) / 8;
    }

    int blockSize = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;

    scalar maxNorm = -1e20;

    for (int it = 0; it < niter; ++it) {

        scalar norm = 0.0;
        cudaMemcpy(devNorm, &norm, sizeof(scalar), cudaMemcpyHostToDevice);

        GaussSeidelIterateKernel<<<numBlocks, threadsPerBlock, sizeof(scalar) * blockSize>>>(devTemp, devCoef, devNorm);
        
        cudaMemcpy(&norm, devNorm, sizeof(scalar), cudaMemcpyDeviceToHost);

        norm = sqrt(norm / (nx * ny * nz));

        maxNorm = max(norm, maxNorm);

        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }

    cudaMemcpy(temp.data(), devTemp, tempSize, cudaMemcpyDeviceToHost);

    cudaFree(devTemp);
    cudaFree(devCoef);
    cudaFree(devNorm);
}