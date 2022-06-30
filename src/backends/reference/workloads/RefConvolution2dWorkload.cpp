//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefConvolution2dWorkload.hpp"

#include "ConvImpl.hpp"
#include "RefWorkloadUtils.hpp"

#include "Profiling.hpp"

namespace armnn
{
RefConvolution2dWorkload::RefConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
                                                   const WorkloadInfo& info)
    : RefBaseWorkload<Convolution2dQueueDescriptor>(descriptor, info)
    , m_InputShape(info.m_InputTensorInfos[0].GetShape())
    , m_FilterShape(info.m_InputTensorInfos[1].GetShape())
    , m_OutputShape(info.m_OutputTensorInfos[0].GetShape())
{
    WorkloadInfo detailsInfo;
    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("RefConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());
}

void RefConvolution2dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvolution2dWorkload::ExecuteAsync(ExecutionData& executionData)
{
    WorkingMemDescriptor* workingMemDescriptor = static_cast<WorkingMemDescriptor*>(executionData.m_Data);
    Execute(workingMemDescriptor->m_Inputs, workingMemDescriptor->m_Outputs);
}

void RefConvolution2dWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_GUID(Compute::CpuRef, "RefConvolution2dWorkload_Execute", this->GetGuid());

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());

    std::unique_ptr<Decoder<float>> weightsDecoder = MakeDecoder<float>(GetTensorInfo(inputs[1]), inputs[1]->Map());
    std::unique_ptr<Decoder<float>> biasDecoder;

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        biasDecoder = MakeDecoder<float>(GetTensorInfo(inputs[2]), inputs[2]->Map());
    }

    Convolve(m_InputShape, *inputDecoder, m_OutputShape, *outputEncoder, m_FilterShape,
             *weightsDecoder, m_Data.m_Parameters.m_BiasEnabled, biasDecoder.get(),
             m_Data.m_Parameters.m_DataLayout, m_Data.m_Parameters.m_PadTop, m_Data.m_Parameters.m_PadLeft,
             m_Data.m_Parameters.m_StrideX, m_Data.m_Parameters.m_StrideY,
             m_Data.m_Parameters.m_DilationX, m_Data.m_Parameters.m_DilationY);
}

} //namespace armnn