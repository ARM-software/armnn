//
// Copyright © 2017 Arm Ltd. All rights reserved.
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

void RefConvolution2dWorkload::PostAllocationConfigure()
{
    PostAllocationConfigure(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvolution2dWorkload::PostAllocationConfigure(std::vector<ITensorHandle*> inputs,
                                                        std::vector<ITensorHandle*> outputs)
{
    const TensorInfo& inputInfo = GetTensorInfo(inputs[0]);
    ARMNN_ASSERT(inputInfo.GetNumDimensions() > 1);
    m_InputShape = inputInfo.GetShape();

    const TensorInfo& rFilterInfo = GetTensorInfo(inputs[1]);
    ARMNN_ASSERT(inputInfo.GetNumDimensions() > 1);
    m_FilterShape = rFilterInfo.GetShape();
    m_FilterDecoder = MakeDecoder<float>(rFilterInfo);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        const TensorInfo& biasInfo = GetTensorInfo(inputs[2]);
        m_BiasDecoder = MakeDecoder<float>(biasInfo);
    }

    const TensorInfo& outputInfo = GetTensorInfo(outputs[0]);
    m_OutputShape = outputInfo.GetShape();
}

void RefConvolution2dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvolution2dWorkload::ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)
{
    PostAllocationConfigure(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);

    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefConvolution2dWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_GUID(Compute::CpuRef, "RefConvolution2dWorkload_Execute", this->GetGuid());

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());

    m_FilterDecoder->Reset(inputs[1]->Map());
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasDecoder->Reset(inputs[2]->Map());
    }

    Convolve(m_InputShape, *inputDecoder, m_OutputShape, *outputEncoder, m_FilterShape,
             *m_FilterDecoder, m_Data.m_Parameters.m_BiasEnabled, m_BiasDecoder.get(),
             m_Data.m_Parameters.m_DataLayout, m_Data.m_Parameters.m_PadTop, m_Data.m_Parameters.m_PadLeft,
             m_Data.m_Parameters.m_StrideX, m_Data.m_Parameters.m_StrideY,
             m_Data.m_Parameters.m_DilationX, m_Data.m_Parameters.m_DilationY);
}

} //namespace armnn