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
        : RefBaseWorkload<FullyConnectedQueueDescriptor>(descriptor, info)
{
}

void RefFullyConnectedWorkload::PostAllocationConfigure()
{
    PostAllocationConfigure(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefFullyConnectedWorkload::PostAllocationConfigure(std::vector<ITensorHandle*> inputs,
                                                        std::vector<ITensorHandle*> outputs)
{
    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    ARMNN_ASSERT(inputInfo.GetNumDimensions() > 1);
    m_InputShape = inputInfo.GetShape();

    const TensorInfo& rWeightInfo = GetTensorInfo(inputs[1]);
    ARMNN_ASSERT(inputInfo.GetNumDimensions() > 1);
    m_WeightShape = rWeightInfo.GetShape();
    m_WeightDecoder = MakeDecoder<float>(rWeightInfo);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        const TensorInfo& biasInfo = GetTensorInfo(inputs[2]);
        m_BiasDecoder = MakeDecoder<float>(biasInfo);
    }

    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);
    m_OutputShape = outputInfo.GetShape();

    m_NumActivations = 1; // Total number of activations in the input.
    for (unsigned int i = 1; i < inputInfo.GetNumDimensions(); i++)
    {
        m_NumActivations *= inputInfo.GetShape()[i];
    }
}

void RefFullyConnectedWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefFullyConnectedWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    PostAllocationConfigure(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);

    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefFullyConnectedWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefFullyConnectedWorkload_Execute");

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> OutputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());

    m_WeightDecoder->Reset(inputs[1]->Map());
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasDecoder->Reset(inputs[2]->Map());
    }

    FullyConnected(m_InputShape,
                   *inputDecoder,
                   m_OutputShape,
                   *OutputEncoder,
                   m_WeightShape,
                   *m_WeightDecoder,
                   m_BiasDecoder.get(),
                   m_Data.m_Parameters.m_BiasEnabled,
                   m_NumActivations,
                   m_Data.m_Parameters.m_TransposeWeightMatrix);
}

} //namespace armnn
