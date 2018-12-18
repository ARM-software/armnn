//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDepthwiseConvolutionWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <neon/NeonLayerSupport.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadUtils.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
                                                             const TensorInfo& output,
                                                             const DepthwiseConvolution2dDescriptor& descriptor,
                                                             const TensorInfo& weights,
                                                             const Optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input,  descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    // ArmNN's weight format is [ M, I, H, W ]
    const unsigned int aclDepthMultiplier = weights.GetShape()[0];

    // Convert the weight format from ArmNN's [ M, I, H, W ] (does NOT depend on the data layout) to either
    // [ 1, H, W, I * M ] (if NHWC) or [ 1, I * M, H, W ] (if NCHW), as required by the compute library
    TensorInfo weightsPermuted = ConvertWeightTensorInfoFromArmnnToAcl(weights, descriptor.m_DataLayout);

    // Convert the weights into the compute library format
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weightsPermuted, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        BOOST_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    const arm_compute::PadStrideInfo aclPadStrideInfo = BuildArmComputePadStrideInfo(descriptor);

    return arm_compute::NEDepthwiseConvolutionLayer::validate(&aclInputInfo,
                                                              &aclWeightsInfo,
                                                              optionalAclBiasesInfo,
                                                              &aclOutputInfo,
                                                              aclPadStrideInfo,
                                                              aclDepthMultiplier);
}

NeonDepthwiseConvolutionWorkload::NeonDepthwiseConvolutionWorkload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : BaseWorkload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    // ArmNN's weight format is [ M, I, H, W ]
    auto& weightInfo = m_Data.m_Weight->GetTensorInfo();

    // Allocate a buffer for the swizzling of the weight tensor
    std::unique_ptr<unsigned char[]> permuteBuffer(new unsigned char[m_Data.m_Weight->GetTensorInfo().GetNumBytes()]);

    // Convert the weight format from ArmNN's [ M, I, H, W ] (does NOT depend on the data layout) to either
    // [ 1, H, W, I * M ] (if NHWC) or [ 1, I * M, H, W ] (if NCHW), as required by the compute library
    ConstTensor weightPermuted = ConvertWeightTensorFromArmnnToAcl(m_Data.m_Weight,
                                                                   m_Data.m_Parameters.m_DataLayout,
                                                                   permuteBuffer.get());

    // Convert the weights into the compute library format
    m_KernelTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_KernelTensor, weightPermuted.GetInfo(), m_Data.m_Parameters.m_DataLayout);

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

    m_Data.ValidateInputsOutputs("NeonDepthwiseConvolutionWorkload", 1, 1);

    INeonTensorHandle* inputTensorHandle  = static_cast<INeonTensorHandle*>(m_Data.m_Inputs[0]);
    INeonTensorHandle* outputTensorHandle = static_cast<INeonTensorHandle*>(m_Data.m_Outputs[0]);

    arm_compute::ITensor& input  = inputTensorHandle->GetTensor();
    arm_compute::ITensor& output = outputTensorHandle->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    // Get the depth multiplier
    const unsigned int depthMultiplier = weightInfo.GetShape()[0];

    // Check for optimisation opportunities.
    bool use3x3Optimisation = (weightInfo.GetShape()[2] == 3) && (weightInfo.GetShape()[3] == 3);
    if (use3x3Optimisation)
    {
        m_pDepthwiseConvolutionLayer = std::make_unique<arm_compute::NEDepthwiseConvolutionLayer3x3>();
        static_cast<arm_compute::NEDepthwiseConvolutionLayer3x3*>(
            m_pDepthwiseConvolutionLayer.get())->configure(&input,
                                                           m_KernelTensor.get(),
                                                           m_BiasTensor.get(),
                                                           &output,
                                                           padStrideInfo,
                                                           depthMultiplier);
    }
    else
    {
        m_pDepthwiseConvolutionLayer = std::make_unique<arm_compute::NEDepthwiseConvolutionLayer>();
        static_cast<arm_compute::NEDepthwiseConvolutionLayer*>(
            m_pDepthwiseConvolutionLayer.get())->configure(&input,
                                                           m_KernelTensor.get(),
                                                           m_BiasTensor.get(),
                                                           &output,
                                                           padStrideInfo,
                                                           depthMultiplier);
    }

    BOOST_ASSERT(m_pDepthwiseConvolutionLayer);

    ScopedCpuTensorHandle weightsPermutedHandle(weightPermuted);
    InitializeArmComputeTensorData(*m_KernelTensor, &weightsPermutedHandle);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        InitializeArmComputeTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    m_pDepthwiseConvolutionLayer->prepare();
    FreeUnusedTensors();
}

void NeonDepthwiseConvolutionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonDepthwiseConvolutionWorkload_Execute");
    BOOST_ASSERT(m_pDepthwiseConvolutionLayer);

    m_pDepthwiseConvolutionLayer->run();
}

void NeonDepthwiseConvolutionWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

} //namespace armnn
