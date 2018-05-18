//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonConvolution2dUint8Workload.hpp"

namespace armnn
{

NeonConvolution2dUint8Workload::NeonConvolution2dUint8Workload(const Convolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : NeonConvolution2dBaseWorkload(descriptor, info, memoryManager)
{
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        InitialiseArmComputeTensorData(m_BiasTensor, m_Data.m_Bias->template GetConstTensor<int32_t>());
    }
}


void NeonConvolution2dUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonConvolution2dUint8Workload_Execute");
    m_ConvolutionLayer->run();
}

void NeonConvolution2dUint8Workload::ValidateData() const
{
    m_Data.ValidateInputsOutputs("NeonConvolution2dUint8Workload", 1, 1);
}

} //namespace armnn
