//
// Copyright © 2019 Arm Ltd. All rights reserved.
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
        : BaseWorkload<ArgMinMaxQueueDescriptor>(descriptor, info) {}

void RefArgMinMaxWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT(Compute::CpuRef, "RefArgMinMaxWorkload_Execute");

    const TensorInfo &inputTensorInfo = GetTensorInfo(m_Data.m_Inputs[0]);

    std::unique_ptr<Decoder<float>> decoderPtr = MakeDecoder<float>(inputTensorInfo, m_Data.m_Inputs[0]->Map());
    Decoder<float> &decoder = *decoderPtr;

    const TensorInfo &outputTensorInfo = GetTensorInfo(m_Data.m_Outputs[0]);

    if (m_Data.m_Parameters.m_Output_Type == armnn::DataType::Signed32) {
        int32_t *output = GetOutputTensorData<int32_t>(0, m_Data);
        ArgMinMax(decoder, output, inputTensorInfo, outputTensorInfo, m_Data.m_Parameters.m_Function,
                  m_Data.m_Parameters.m_Axis);
    } else {
        int64_t *output = GetOutputTensorData<int64_t>(0, m_Data);
        ArgMinMax(decoder, output, inputTensorInfo, outputTensorInfo, m_Data.m_Parameters.m_Function,
                  m_Data.m_Parameters.m_Axis);
    }
}

} //namespace armnn