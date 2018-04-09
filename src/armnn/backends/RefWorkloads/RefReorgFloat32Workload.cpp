
#include "RefReorgFloat32Workload.hpp"

#include "ConvImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{
    template<typename Dtype>
    void reorg_cpu(const Dtype *x, unsigned int w, unsigned int h, unsigned int c, unsigned int batch, unsigned int stride, int forward, Dtype *out)
    {
        unsigned int b,i,j,k;
        unsigned int out_c = c/(stride*stride);

        for(b = 0; b < batch; ++b){
            for(k = 0; k < c; ++k){
                for(j = 0; j < h; ++j){
                    for(i = 0; i < w; ++i){
                        unsigned int in_index  = i + w*(j + h*(k + c*b));
                        unsigned int c2 = k % out_c;
                        unsigned int offset = k / out_c;
                        unsigned int w2 = i*stride + offset % stride;
                        unsigned int h2 = j*stride + offset / stride;
                        unsigned int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                        if(forward) out[out_index] = x[in_index];
                        else out[in_index] = x[out_index];
                    }
                }
            }
        }
    }



    void RefReorgFloat32Workload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefReorgFloat32Workload_Execute");

        float*       outputData = GetOutputTensorDataFloat(0, m_Data);
        const float* inputData  = GetInputTensorDataFloat(0, m_Data);

        TensorInfo info = GetTensorInfo(m_Data.m_Inputs[0]);
        TensorShape shape = info.GetShape();
        unsigned int stride = m_Data.m_Parameters.m_Stride;

        reorg_cpu<float>(inputData,shape[2],shape[3],shape[1],shape[0],stride,1,outputData);
    }

} //namespace armnn
