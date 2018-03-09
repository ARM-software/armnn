//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//

#include "RefFullyConnectedFloat32Workload.hpp"

#include "FullyConnected.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

void RefFullyConnectedFloat32Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFullyConnectedFloat32Workload_Execute");

    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    float*       outputData = GetOutputTensorDataFloat(0, m_Data);
    const float* inputData  = GetInputTensorDataFloat(0, m_Data);
    const float* weightData = m_Data.m_Weight->GetConstTensor<float>();
    const float* biasData   = m_Data.m_Parameters.m_BiasEnabled ? m_Data.m_Bias->GetConstTensor<float>() : nullptr;

    FullyConnected(inputData,
                   outputData,
                   inputInfo,
                   outputInfo,
                   weightData,
                   biasData,
                   m_Data.m_Parameters.m_TransposeWeightMatrix);
}

} //namespace armnn
