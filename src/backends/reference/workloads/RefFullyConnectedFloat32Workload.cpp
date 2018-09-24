//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFullyConnectedFloat32Workload.hpp"

#include "FullyConnected.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{
RefFullyConnectedFloat32Workload::RefFullyConnectedFloat32Workload(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info)
        : Float32Workload<FullyConnectedQueueDescriptor>(descriptor, info),
          m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight))),
          m_Bias(descriptor.m_Parameters.m_BiasEnabled
                 ? std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias)) : nullptr) {}

void RefFullyConnectedFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFullyConnectedFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    float*       outputData = GetOutputTensorDataFloat(0, m_Data);
    const float* inputData  = GetInputTensorDataFloat(0, m_Data);
    const float* weightData = m_Weight->GetConstTensor<float>();
    const float* biasData   = m_Data.m_Parameters.m_BiasEnabled ? m_Bias->GetConstTensor<float>() : nullptr;

    FullyConnected(inputData,
                   outputData,
                   inputInfo,
                   outputInfo,
                   weightData,
                   biasData,
                   m_Data.m_Parameters.m_TransposeWeightMatrix);
}

} //namespace armnn
