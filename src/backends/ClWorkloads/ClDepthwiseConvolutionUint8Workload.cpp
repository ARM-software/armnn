//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDepthwiseConvolutionUint8Workload.hpp"

#include "backends/CpuTensorHandle.hpp"

#include "ClWorkloadUtils.hpp"

namespace armnn
{

ClDepthwiseConvolutionUint8Workload::ClDepthwiseConvolutionUint8Workload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : ClDepthwiseConvolutionBaseWorkload(descriptor, info)
{
    InitializeArmComputeClTensorData(*m_KernelTensor, m_Data.m_Weight);

    if (m_BiasTensor)
    {
        InitializeArmComputeClTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    m_DepthwiseConvolutionLayer->prepare();
    FreeUnusedTensors();
}

void ClDepthwiseConvolutionUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClDepthwiseConvolutionUint8Workload_Execute");
    BOOST_ASSERT(m_DepthwiseConvolutionLayer);

    m_DepthwiseConvolutionLayer->run();
}

} //namespace armnn

