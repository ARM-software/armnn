//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvolution2dFloatWorkload.hpp"
#include <backends/CpuTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>
#include <backends/neon/NeonLayerSupport.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

NeonConvolution2dFloatWorkload::NeonConvolution2dFloatWorkload(const Convolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : NeonConvolution2dBaseWorkload(descriptor, info, memoryManager)
{
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        InitializeArmComputeTensorDataForFloatTypes(*m_BiasTensor, m_Data.m_Bias);
    }

    m_ConvolutionLayer->prepare();
    FreeUnusedTensors();
}

void NeonConvolution2dFloatWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConvolution2dFloatWorkload_Execute");
    m_ConvolutionLayer->run();
}

void NeonConvolution2dFloatWorkload::ValidateData() const
{
    m_Data.ValidateInputsOutputs("NeonConvolution2dFloatWorkload", 1, 1);
}

} //namespace armnn

