
#include "RefDetectionOutputFloat32Workload.hpp"

#include "ConvImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

    void RefDetectionOutputFloat32Workload::Execute() const
    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDetectionOutputFloat32Workload_Execute");

        //float*       outputData = GetOutputTensorDataFloat(0, m_Data);
        //const float* inputData  = GetInputTensorDataFloat(0, m_Data);
        //const float* weightData = m_Data.m_Weight->template GetConstTensor<float>();
        //const float* biasData   = m_Data.m_Parameters.m_BiasEnabled ?
        //                          m_Data.m_Bias->template GetConstTensor<float>() : nullptr;

        //ConvImpl<armnn::DepthwiseConvolution2dQueueDescriptor, float, float, float>
        //        (m_Data, inputData, 0.0f, 0, weightData, 0.0f, 0, biasData, outputData, 0.0f, 0, true);
        // todo
    }

} //namespace armnn
