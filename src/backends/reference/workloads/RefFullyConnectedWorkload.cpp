//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFullyConnectedWorkload.hpp"

#include "FullyConnected.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{
RefFullyConnectedWorkload::RefFullyConnectedWorkload(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info)
        : BaseWorkload<FullyConnectedQueueDescriptor>(descriptor, info),
          m_Weight(std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Weight)))
{
    const TensorInfo& rWeightInfo = m_Weight->GetTensorInfo();
    m_WeightShape = rWeightInfo.GetShape();
    m_WeightDecoder = MakeDecoder<float>(rWeightInfo, m_Weight->Map(true));

    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        m_Bias = std::make_unique<ScopedCpuTensorHandle>(*(descriptor.m_Bias));
        const TensorInfo& biasInfo = m_Bias->GetTensorInfo();
        m_BiasDecoder = MakeDecoder<float>(biasInfo, m_Bias->Map(true));
    }
}

void RefFullyConnectedWorkload::PostAllocationConfigure()
{
    const TensorInfo& inputInfo = GetTensorInfo(m_Data.m_Inputs[0]);
    ARMNN_ASSERT(inputInfo.GetNumDimensions() > 1);
    m_InputShape = inputInfo.GetShape();
    m_InputDecoder = MakeDecoder<float>(inputInfo);

    const TensorInfo& outputInfo = GetTensorInfo(m_Data.m_Outputs[0]);
    m_OutputShape = outputInfo.GetShape();
    m_OutputEncoder = MakeEncoder<float>(outputInfo);

    m_NumActivations = 1; // Total number of activations in the input.
    for (unsigned int i = 1; i < inputInfo.GetNumDimensions(); i++)
    {
        m_NumActivations *= inputInfo.GetShape()[i];
    }
}

void RefFullyConnectedWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFullyConnectedWorkload_Execute");

    m_InputDecoder->Reset(m_Data.m_Inputs[0]->Map());
    m_OutputEncoder->Reset(m_Data.m_Outputs[0]->Map());

    FullyConnected(m_InputShape,
                   *m_InputDecoder,
                   m_OutputShape,
                   *m_OutputEncoder,
                   m_WeightShape,
                   *m_WeightDecoder,
                   *m_BiasDecoder,
                   m_Data.m_Parameters.m_BiasEnabled,
                   m_NumActivations,
                   m_Data.m_Parameters.m_TransposeWeightMatrix);
}

} //namespace armnn
