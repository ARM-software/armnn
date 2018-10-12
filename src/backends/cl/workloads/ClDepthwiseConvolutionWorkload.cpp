//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDepthwiseConvolutionWorkload.hpp"

#include "TypeUtils.hpp"
#include "ClWorkloadUtils.hpp"

#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/CpuTensorHandle.hpp>

#include <arm_compute/runtime/CL/functions/CLDepthwiseConvolutionLayer.h>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const DepthwiseConvolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const Optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        BOOST_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    const arm_compute::PadStrideInfo aclPadStrideInfo = BuildArmComputePadStrideInfo(descriptor);
    const unsigned int aclDepthMultiplier = weights.GetShape()[0];

    return arm_compute::CLDepthwiseConvolutionLayer::validate(&aclInputInfo,
                                                              &aclWeightsInfo,
                                                              optionalAclBiasesInfo,
                                                              &aclOutputInfo,
                                                              aclPadStrideInfo,
                                                              aclDepthMultiplier);
}

ClDepthwiseConvolutionWorkload::ClDepthwiseConvolutionWorkload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : BaseWorkload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    auto& weightInfo = m_Data.m_Weight->GetTensorInfo();

    m_KernelTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_KernelTensor, weightInfo, m_Data.m_Parameters.m_DataLayout);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    arm_compute::PadStrideInfo padStrideInfo(m_Data.m_Parameters.m_StrideX,
                                             m_Data.m_Parameters.m_StrideY,
                                             m_Data.m_Parameters.m_PadLeft,
                                             m_Data.m_Parameters.m_PadRight,
                                             m_Data.m_Parameters.m_PadTop,
                                             m_Data.m_Parameters.m_PadBottom,
                                             arm_compute::DimensionRoundingType::FLOOR);

    std::string name = std::string("ClDepthwiseConvolutionWorkload");
    m_Data.ValidateInputsOutputs(name, 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    const unsigned int depthMultiplier = weightInfo.GetShape()[0];

    //Check for optimisation opportunities.
    bool use3x3Optimisation = (weightInfo.GetShape()[3] == 3) && (weightInfo.GetShape()[2] == 3);
    if (use3x3Optimisation)
    {
        m_DepthwiseConvolutionLayer = std::make_unique<arm_compute::CLDepthwiseConvolutionLayer3x3>();
        static_cast<arm_compute::CLDepthwiseConvolutionLayer3x3*>(m_DepthwiseConvolutionLayer.get())->configure(
            &input,
            m_KernelTensor.get(),
            m_BiasTensor.get(),
            &output,
            padStrideInfo,
            depthMultiplier);
    }
    else
    {
        m_DepthwiseConvolutionLayer = std::make_unique<arm_compute::CLDepthwiseConvolutionLayer>();
        static_cast<arm_compute::CLDepthwiseConvolutionLayer*>(m_DepthwiseConvolutionLayer.get())->configure(
            &input,
            m_KernelTensor.get(),
            m_BiasTensor.get(),
            &output,
            padStrideInfo,
            depthMultiplier);
    }

    BOOST_ASSERT(m_DepthwiseConvolutionLayer);

    InitializeArmComputeClTensorData(*m_KernelTensor, m_Data.m_Weight);

    if (m_BiasTensor)
    {
        InitializeArmComputeClTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    m_DepthwiseConvolutionLayer->prepare();
    FreeUnusedTensors();
}

void ClDepthwiseConvolutionWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

void ClDepthwiseConvolutionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClDepthwiseConvolutionWorkload_Execute");
    BOOST_ASSERT(m_DepthwiseConvolutionLayer);

    m_DepthwiseConvolutionLayer->run();
}

} // namespace armnn
