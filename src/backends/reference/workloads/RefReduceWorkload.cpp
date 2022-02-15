//
// Copyright © 2020 Samsung Electronics Co Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefReduceWorkload.hpp"

#include "Reduce.hpp"
#include "RefWorkloadUtils.hpp"
#include "BaseIterator.hpp"
#include "Profiling.hpp"

namespace armnn
{

RefReduceWorkload::RefReduceWorkload(
    const ReduceQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : RefBaseWorkload<ReduceQueueDescriptor>(descriptor, info) {}

void RefReduceWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefReduceWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefReduceWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefReduceWorkload_Execute");

    const TensorInfo& inputInfo  = GetTensorInfo(inputs[0]);
    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputInfo, inputs[0]->Map());
    Decoder<float>& decoder = *decoderPtr;

    std::unique_ptr<Encoder<float>> encoderPtr = MakeEncoder<float>(outputInfo, outputs[0]->Map());
    Encoder<float>& encoder = *encoderPtr;

    Reduce(inputInfo,
           outputInfo,
           decoder,
           encoder,
           m_Data.m_Parameters.m_vAxis,
           m_Data.m_Parameters.m_ReduceOperation);
}

} //namespace armnn
