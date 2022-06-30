//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefFullyConnectedWorkload.hpp"

#include "FullyConnected.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{

unsigned int GetNumActivations(const TensorInfo& inputInfo)
{
    unsigned int numActivations = 1; // Total number of activations in the input.
    for (unsigned int i = 1; i < inputInfo.GetNumDimensions(); i++)
    {
        numActivations *= inputInfo.GetShape()[i];
    }
    return numActivations;
}


RefFullyConnectedWorkload::RefFullyConnectedWorkload(
    const FullyConnectedQueueDescriptor& descriptor, const WorkloadInfo& info)
        : RefBaseWorkload<FullyConnectedQueueDescriptor>(descriptor, info)
        , m_InputShape(info.m_InputTensorInfos[0].GetShape())
        , m_WeightShape(info.m_InputTensorInfos[1].GetShape())
        , m_OutputShape(info.m_OutputTensorInfos[0].GetShape())
        , m_NumActivations(GetNumActivations(info.m_InputTensorInfos[0]))
{
}

void RefFullyConnectedWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefFullyConnectedWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefFullyConnectedWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFullyConnectedWorkload_Execute");

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> OutputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());

    std::unique_ptr<Decoder<float>> weightsDecoder = MakeDecoder<float>(GetTensorInfo(inputs[1]), inputs[1]->Map());
    std::unique_ptr<Decoder<float>> biasDecoder;

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        biasDecoder = MakeDecoder<float>(GetTensorInfo(inputs[2]), inputs[2]->Map());
    }

    FullyConnected(m_InputShape,
                   *inputDecoder,
                   m_OutputShape,
                   *OutputEncoder,
                   m_WeightShape,
                   *weightsDecoder,
                   biasDecoder.get(),
                   m_Data.m_Parameters.m_BiasEnabled,
                   m_NumActivations,
                   m_Data.m_Parameters.m_TransposeWeightMatrix);
}

} //namespace armnn
