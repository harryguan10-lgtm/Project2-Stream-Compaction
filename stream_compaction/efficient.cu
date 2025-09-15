#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int* data, int stride) {
            int index = (blockIdx.x * blockDim.x + threadIdx.x) * (stride << 1);

            if (index + stride < n) {
                data[index + (stride << 1) - 1] += data[index + stride - 1];
            }
        }

      
        __global__ void downSweep(int n, int* data, int stride) {
            int index = (blockIdx.x * blockDim.x + threadIdx.x) * (stride << 1);

            if (index + stride < n) {
                int temp = data[index + stride - 1];
                data[index + stride - 1] = data[index + (stride << 1) - 1];
                data[index + (stride << 1) - 1] += temp;
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int pow2_n = 1 << ilog2ceil(n);

            int* dev_data;
            cudaMalloc((void**)&dev_data, pow2_n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to device failed!");

            if (pow2_n > n) {
                cudaMemset(dev_data + n, 0, (pow2_n - n) * sizeof(int));
                checkCUDAError("cudaMemset failed!");
            }

            const int blockSize = 128;

            for (int stride = 1; stride < pow2_n; stride <<= 1) {
                int numThreads = pow2_n >> (ilog2(stride) + 1);
                if (numThreads > 0) {
                    dim3 blockDim(blockSize);
                    dim3 gridDim((numThreads + blockSize - 1) / blockSize);

                    upSweep << <gridDim, blockDim >> > (pow2_n, dev_data, stride);
                    checkCUDAError("upSweep failed!");
                }
            }

            cudaMemset(dev_data + pow2_n - 1, 0, sizeof(int));
            checkCUDAError("cudaMemset last element failed!");

            for (int stride = pow2_n >> 1; stride > 0; stride >>= 1) {
                int numThreads = pow2_n >> (ilog2(stride) + 1);
                if (numThreads > 0) {
                    dim3 blockDim(blockSize);
                    dim3 gridDim((numThreads + blockSize - 1) / blockSize);

                    downSweep << <gridDim, blockDim >> > (pow2_n, dev_data, stride);
                    checkCUDAError("downS failed!");
                }
            }

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from device failed!");

            cudaFree(dev_data);

            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
