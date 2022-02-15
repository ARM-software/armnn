//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefArgMinMaxWorkload.hpp"

#include "ArgMinMax.hpp"
#include "RefWorkloadUtils.hpp"
#include "Decoders.hpp"
#include "Encoders.hpp"
#include "Profiling.hpp"

namespace armnn
{
RefArgMinMaxWorkload::RefArgMinMaxWorkload(
        const ArgMinMaxQueueDescriptor& descriptor,
        const WorkloadInfo& info)
        : RefBaseWorkload<ArgMinMaxQueueDescriptor>(descriptor, info) {}


void RefArgMinMaxWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefArgMinMaxWorkload::ExecuteAsync(WorkingMemDescriptor &workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefArgMinMaxWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefArgMinMaxWorkload_Execute");

    const TensorInfo &inputTensorInfo = GetTensorInfo(inputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputTensorInfo, inputs[0]->Map());
    Decoder<float> &decoder = *decoderPtr;

    const TensorInfo &outputTensorInfo = GetTensorInfo(outputs[0]);

    if (outputTensorInfo.GetDataType() == armnn::DataType::Signed32) {
        int32_t *output = GetOutputTensorData<int32_t>(outputs[0]);
        ArgMinMax(decoder, output, inputTensorInfo, outputTensorInfo, m_Data.m_Parameters.m_Function,
                  m_Data.m_Parameters.m_Axis);
    } else {
        int64_t *output = GetOutputTensorData<int64_t>(outputs[0]);
        ArgMinMax(decoder, output, inputTensorInfo, outputTensorInfo, m_Data.m_Parameters.m_Function,
                  m_Data.m_Parameters.m_Axis);
    }
}

} //namespace armnn