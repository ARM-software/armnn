//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "ClDepthwiseConvolutionFloat32Workload.hpp"

#include "backends/ClWorkloadUtils.hpp"
#include "backends/CpuTensorHandle.hpp"

namespace armnn
{

ClDepthwiseConvolutionFloat32Workload::ClDepthwiseConvolutionFloat32Workload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : ClDepthwiseConvolutionBaseWorkload(descriptor, info)
{
    InitializeArmComputeClTensorDataForFloatTypes(*m_KernelTensor, m_Data.m_Weight);

    if (m_BiasTensor)
    {
        InitializeArmComputeClTensorDataForFloatTypes(*m_BiasTensor, m_Data.m_Bias);
    }

    m_DepthwiseConvolutionLayer->prepare();
    FreeUnusedTensors();
}

void ClDepthwiseConvolutionFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClDepthwiseConvolutionFloat32Workload_Execute");
    BOOST_ASSERT(m_DepthwiseConvolutionLayer);

    m_DepthwiseConvolutionLayer->run();
}

} //namespace armnn
