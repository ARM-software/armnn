//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefQuantizeWorkload.hpp"

#include "RefWorkloadUtils.hpp"

#include <armnn/TypesUtils.hpp>


namespace armnn
{

namespace
{

void QuantizeImpl(Decoder<float>& in, Encoder<float>& out, size_t numValues)
{
    for (unsigned int i = 0; i < numValues; i++)
    {
        in[i];
        out[i];
        out.Set(in.Get());
    }
}

} //namespace

RefQuantizeWorkload::RefQuantizeWorkload(const QuantizeQueueDescriptor& descriptor, const WorkloadInfo &info)
    : RefBaseWorkload(descriptor, info)
    , m_NumElements(info.m_InputTensorInfos[0].GetNumElements())
{
}

void RefQuantizeWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefQuantizeWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefQuantizeWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());

    QuantizeImpl(*inputDecoder, *outputEncoder, m_NumElements);
}

} //namespace armnn