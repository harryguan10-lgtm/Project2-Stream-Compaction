#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernNaiveScan(int n, int d, int* odata, const int* idata) {
            int k = blockIdx.x * blockDim.x + threadIdx.x;

            if (k >= n) return;

            int offset = 1 << (d - 1);

            if (k >= offset) {
                odata[k] = idata[k] + idata[k - offset];
            }
            else {
                odata[k] = idata[k];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();

            int* dev_in, * dev_out;
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 128;
            dim3 blockDim(blockSize);
            dim3 gridDim((n + blockSize - 1) / blockSize);

            for (int d = 1; d <= ilog2ceil(n); d++) {
                kernNaiveScan << <gridDim, blockDim >> > (n, d, dev_out, dev_in);

                int* temp = dev_in;
                dev_in = dev_out;
                dev_out = temp;
            }


            cudaMemcpy(odata + 1, dev_in, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            odata[0] = 0; 
            cudaFree(dev_in);
            cudaFree(dev_out);
            timer().endGpuTimer();
        }
    }
}
