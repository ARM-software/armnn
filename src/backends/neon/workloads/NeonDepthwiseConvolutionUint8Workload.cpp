//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDepthwiseConvolutionUint8Workload.hpp"
#include <backends/neon/NeonLayerSupport.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>

namespace armnn
{
using namespace armcomputetensorutils;

NeonDepthwiseConvolutionUint8Workload::NeonDepthwiseConvolutionUint8Workload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : Uint8Workload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    const TensorInfo& weightInfo = m_Data.m_Weight->GetTensorInfo();

    m_KernelTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_KernelTensor, weightInfo, m_Data.m_Parameters.m_DataLayout);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    arm_compute::PadStrideInfo padStrideInfo(m_Data.m_Parameters.m_StrideX,
                                             m_Data.m_Parameters.m_StrideY,
                                             m_Data.m_Parameters.m_PadLeft,
                                             m_Data.m_Parameters.m_PadRight,
                                             m_Data.m_Parameters.m_PadTop,
                                             m_Data.m_Parameters.m_PadBottom,
                                             arm_compute::DimensionRoundingType::FLOOR);

    m_Data.ValidateInputsOutputs("NeonDepthwiseConvolutionUint8Workload", 1, 1);

    arm_compute::ITensor& input  = static_cast<INeonTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = static_cast<INeonTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    bool use3x3Optimisation = weightInfo.GetShape()[3] == 3 && weightInfo.GetShape()[2] == 3;
    if (use3x3Optimisation)
    {
        m_pDepthwiseConvolutionLayer = std::make_unique<arm_compute::NEDepthwiseConvolutionLayer3x3>();
        static_cast<arm_compute::NEDepthwiseConvolutionLayer3x3*>(
            m_pDepthwiseConvolutionLayer.get())->configure(&input,
                                                           m_KernelTensor.get(),
                                                           m_BiasTensor.get(),
                                                           &output,
                                                           padStrideInfo);
    }
    else
    {
        m_pDepthwiseConvolutionLayer = std::make_unique<arm_compute::NEDepthwiseConvolutionLayer>();
        static_cast<arm_compute::NEDepthwiseConvolutionLayer*>(
            m_pDepthwiseConvolutionLayer.get())->configure(&input,
                                                           m_KernelTensor.get(),
                                                           m_BiasTensor.get(),
                                                           &output,
                                                           padStrideInfo);
    }

    BOOST_ASSERT(m_pDepthwiseConvolutionLayer);

    InitialiseArmComputeTensorData(*m_KernelTensor, m_Data.m_Weight->GetConstTensor<uint8_t>());

    if (m_BiasTensor)
    {
        InitialiseArmComputeTensorData(*m_BiasTensor, m_Data.m_Bias->GetConstTensor<int32_t>());
    }

    m_pDepthwiseConvolutionLayer->prepare();
    FreeUnusedTensors();
}

void NeonDepthwiseConvolutionUint8Workload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonDepthwiseConvolutionUint8Workload_Execute");
    BOOST_ASSERT(m_pDepthwiseConvolutionLayer);

    m_pDepthwiseConvolutionLayer->run();
}

void NeonDepthwiseConvolutionUint8Workload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

} //namespace armnn
