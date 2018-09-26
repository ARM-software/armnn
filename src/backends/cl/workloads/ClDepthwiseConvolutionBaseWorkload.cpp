//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDepthwiseConvolutionBaseWorkload.hpp"

#include "TypeUtils.hpp"

#include <backends/aclCommon/ArmComputeUtils.hpp>
#include <backends/aclCommon/ArmComputeTensorUtils.hpp>
#include <backends/cl/ClTensorHandle.hpp>
#include <backends/CpuTensorHandle.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
    const TensorInfo& output,
    const DepthwiseConvolution2dDescriptor& descriptor,
    const TensorInfo& weights,
    const boost::optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        BOOST_ASSERT(biases.is_initialized());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.get(), descriptor.m_DataLayout);
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

template<armnn::DataType... dataTypes>
ClDepthwiseConvolutionBaseWorkload<dataTypes...>::ClDepthwiseConvolutionBaseWorkload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : TypedWorkload<DepthwiseConvolution2dQueueDescriptor, dataTypes...>(descriptor, info)
{
    auto& weightInfo = m_Data.m_Weight->GetTensorInfo();

    m_KernelTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_KernelTensor, weightInfo);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo());
    }

    arm_compute::PadStrideInfo padStrideInfo(m_Data.m_Parameters.m_StrideX,
                                             m_Data.m_Parameters.m_StrideY,
                                             m_Data.m_Parameters.m_PadLeft,
                                             m_Data.m_Parameters.m_PadRight,
                                             m_Data.m_Parameters.m_PadTop,
                                             m_Data.m_Parameters.m_PadBottom,
                                             arm_compute::DimensionRoundingType::FLOOR);

    std::string name = std::string("ClDepthwiseConvolution") +
            GetDataTypeName(m_Data.m_Weight->GetTensorInfo().GetDataType()) + "Workload";
    m_Data.ValidateInputsOutputs(name, 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

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
}

template<armnn::DataType... dataTypes>
void ClDepthwiseConvolutionBaseWorkload<dataTypes...>::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

// Generate known implementations for linker
template class ClDepthwiseConvolutionBaseWorkload<DataType::Float16, DataType::Float32>;
template class ClDepthwiseConvolutionBaseWorkload<DataType::QuantisedAsymm8>;

} // namespace armnn
