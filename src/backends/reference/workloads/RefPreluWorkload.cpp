//
// Copyright © 2019,2021-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefPreluWorkload.hpp"

#include "RefWorkloadUtils.hpp"
#include "PreluImpl.hpp"

#include <Profiling.hpp>

namespace armnn
{

RefPreluWorkload::RefPreluWorkload(const PreluQueueDescriptor& descriptor,
                                   const WorkloadInfo& info)
    : RefBaseWorkload(descriptor, info)
{}

void RefPreluWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefPreluWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_REF_NAME_GUID("RefPreluWorkload_Execute");
    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& alphaInfo  = GetTensorInfo(inputs[1]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]),
                                                                      inputs[0]->Map());
    std::unique_ptr<Decoder<float>> alphaDecoder = MakeDecoder<float>(GetTensorInfo(inputs[1]),
                                                                      inputs[1]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]),
                                                                       outputs[0]->Map());

    PreluImpl(inputInfo, alphaInfo, outputInfo, *inputDecoder, *alphaDecoder, *outputEncoder);
}

} // namespace armnn
