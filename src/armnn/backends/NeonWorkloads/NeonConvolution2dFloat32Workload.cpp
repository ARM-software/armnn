//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "NeonConvolution2dFloat32Workload.hpp"
#include "backends/CpuTensorHandle.hpp"
#include "backends/ArmComputeTensorUtils.hpp"
#include "backends/NeonLayerSupport.hpp"

namespace armnn
{
using namespace armcomputetensorutils;

NeonConvolution2dFloat32Workload::NeonConvolution2dFloat32Workload(const Convolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : NeonConvolution2dBaseWorkload(descriptor, info, memoryManager)
{
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        InitialiseArmComputeTensorData(m_BiasTensor, m_Data.m_Bias->template GetConstTensor<float>());
    }
}

void NeonConvolution2dFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuAcc, "NeonConvolution2dFloat32Workload_Execute");
    m_ConvolutionLayer->run();
}

void NeonConvolution2dFloat32Workload::ValidateData() const
{
    m_Data.ValidateInputsOutputs("NeonConvolution2dFloat32Workload", 1, 1);
}

} //namespace armnn

