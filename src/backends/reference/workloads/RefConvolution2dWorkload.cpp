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
RefConvolution2dWorkload::RefConvolution2dWorkload(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info)
    : RefBaseWorkload<Convolution2dQueueDescriptor>(descriptor, info)
{
    WorkloadInfo detailsInfo;
    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Weight->GetTensorInfo());
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Bias->GetTensorInfo());
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("RefConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    m_Weight = std::make_unique<ScopedTensorHandle>(*( descriptor.m_Weight ));
    const TensorInfo& rFilterInfo = m_Weight->GetTensorInfo();

    m_FilterShape = rFilterInfo.GetShape();
    m_FilterDecoder = MakeDecoder<float>(rFilterInfo, m_Weight.get()->Map(true));

    if ( descriptor.m_Parameters.m_BiasEnabled )
    {
        m_Bias = std::make_unique<ScopedTensorHandle>(*( descriptor.m_Bias ));
        const TensorInfo& biasInfo = m_Bias->GetTensorInfo();
        m_BiasDecoder = MakeDecoder<float>(biasInfo, m_Bias->Map(true));
    }
}

void RefConvolution2dWorkload::Execute() const
{
    Execute(m_Data.m_Inputs, m_Data.m_Outputs);
}

void RefConvolution2dWorkload::ExecuteAsync(WorkingMemDescriptor& workingMemDescriptor)
{
    Execute(workingMemDescriptor.m_Inputs, workingMemDescriptor.m_Outputs);
}

void RefConvolution2dWorkload::Execute(std::vector<ITensorHandle*> inputs, std::vector<ITensorHandle*> outputs) const
{
    ARMNN_SCOPED_PROFILING_EVENT_GUID(Compute::CpuRef, "RefConvolution2dWorkload_Execute", this->GetGuid());

    std::unique_ptr<Decoder<float>> inputDecoder = MakeDecoder<float>(GetTensorInfo(inputs[0]), inputs[0]->Map());
    std::unique_ptr<Encoder<float>> outputEncoder = MakeEncoder<float>(GetTensorInfo(outputs[0]), outputs[0]->Map());

    const TensorShape& inputShape = GetTensorInfo(inputs[0]).GetShape();
    const TensorShape& outputShape = GetTensorInfo(outputs[0]).GetShape();

    Convolve(inputShape, *inputDecoder, outputShape, *outputEncoder, m_FilterShape,
             *m_FilterDecoder, m_Data.m_Parameters.m_BiasEnabled, m_BiasDecoder.get(),
             m_Data.m_Parameters.m_DataLayout, m_Data.m_Parameters.m_PadTop, m_Data.m_Parameters.m_PadLeft,
             m_Data.m_Parameters.m_StrideX, m_Data.m_Parameters.m_StrideY,
             m_Data.m_Parameters.m_DilationX, m_Data.m_Parameters.m_DilationY);
}

} //namespace armnn
