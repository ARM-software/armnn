//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefDepthwiseConvolution2dFloat32Workload.hpp"

#include "ConvImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{
RefDepthwiseConvolution2dFloat32Workload::RefDepthwiseConvolution2dFloat32Workload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info)
        : Float32Workload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info),
          m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight))),
          m_Bias(descriptor.m_Parameters.m_BiasEnabled
                 ? std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias)) : nullptr) {}

void RefDepthwiseConvolution2dFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefDepthwiseConvolution2dFloat32Workload_Execute");

    const float* inputData  = GetInputTensorDataFloat(0, m_Data);
    const float* weightData = m_Weight->template GetConstTensor<float>();
    const float* biasData   = m_Data.m_Parameters.m_BiasEnabled ? m_Bias->template GetConstTensor<float>() : nullptr;
    const TensorInfo& filterInfo = m_Weight->GetTensorInfo();

    ConvImpl<armnn::DepthwiseConvolution2dQueueDescriptor, float, float, float>
        (m_Data, inputData, 0.0f, 0, weightData, 0.0f, 0, biasData, 0.0f, 0, filterInfo, true);
}

} //namespace armnn
