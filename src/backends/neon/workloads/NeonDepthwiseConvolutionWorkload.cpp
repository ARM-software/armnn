//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonDepthwiseConvolutionWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>

#include <neon/NeonLayerSupport.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <backendsCommon/WorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h>

using namespace armnnUtils;

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
                                                             const TensorInfo& output,
                                                             const DepthwiseConvolution2dDescriptor& descriptor,
                                                             const TensorInfo& weights,
                                                             const Optional<TensorInfo>& biases,
                                                             const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo  = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    // ArmNN format for weights for depthwise is [1, H, W, C] independently of the input/output layout
    //
    // ACL format for weights for depthwise is:
    // - [1, H, W, C] for [N, H, W, C] input/output layout (matches with ArmNN)
    // - [1, C, H, W] for [N, C, H, W] input/output layout
    //
    // Therefore ArmNN weights have to be permuted when input/output layout is [N, C, H, W] to pass them to ACL.
    // The PermuteDepthwiseConv2dWeights backend optimization takes care of this, but it has not been performed yet,
    // so we do the permute here for the TensorInfo weights.
    unsigned int aclDepthMultiplier;
    TensorInfo weightsPermuted;
    std::tie(weightsPermuted, aclDepthMultiplier) = Convert1HWOTensorInfoToAcl(weights, input, descriptor.m_DataLayout);

    // Convert the weights into the compute library format
    arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weightsPermuted, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo* optionalAclBiasesInfo = nullptr;
    if (descriptor.m_BiasEnabled)
    {
        if(!biases.has_value())
        {
            return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                       "ArmNN NeonDepthwiseConvolutionWorkload has empty bias value."};
        }
        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasesInfo.set_are_values_constant(biases.value().IsConstant());
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    const arm_compute::PadStrideInfo aclPadStrideInfo = BuildArmComputePadStrideInfo(descriptor);
    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                      descriptor.m_DilationY);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
        activationDescriptor);

    return arm_compute::NEDepthwiseConvolutionLayer::validate(&aclInputInfo,
                                                              &aclWeightsInfo,
                                                              optionalAclBiasesInfo,
                                                              &aclOutputInfo,
                                                              aclPadStrideInfo,
                                                              aclDepthMultiplier,
                                                              activationInfo,
                                                              aclDilationInfo);
}

NeonDepthwiseConvolutionWorkload::NeonDepthwiseConvolutionWorkload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info)
    : NeonBaseWorkload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();
    arm_compute::ITensor& weights = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    weights.info()->set_are_values_constant(info.m_InputTensorInfos[1].IsConstant());
    arm_compute::ITensor* biasesPtr = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        biasesPtr = &PolymorphicDowncast<IAclTensorHandle *>(m_Data.m_Inputs[2])->GetTensor();
        biasesPtr->info()->set_are_values_constant(info.m_InputTensorInfos[2].IsConstant());
    }

    arm_compute::TensorShape weightsShape = weights.info()->tensor_shape();
    arm_compute::TensorShape inputShape = input.info()->tensor_shape();

    // The PermuteDepthwiseConv2dWeights backend optimization has been performed,
    // converting weights to have the same data layout as input.
    unsigned int depthMultiplier =
        ComputeDepthwiseConv2dDepthMultiplier(m_Data.m_Parameters.m_DataLayout, weightsShape, inputShape);

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(m_Data.m_Parameters.m_DilationX,
                                                                      m_Data.m_Parameters.m_DilationY);

    uint32_t numInputs = m_Data.m_Parameters.m_BiasEnabled ? 3: 2;
    m_Data.ValidateInputsOutputs("NeonDepthwiseConvolutionWorkload", numInputs, 1);

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    weights.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    const arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    m_pDepthwiseConvolutionLayer = std::make_unique<arm_compute::NEDepthwiseConvolutionLayer>();
    static_cast<arm_compute::NEDepthwiseConvolutionLayer*>(
        m_pDepthwiseConvolutionLayer.get())->configure(&input,
                                                       &weights,
                                                       biasesPtr,
                                                       &output,
                                                       padStrideInfo,
                                                       depthMultiplier,
                                                       activationInfo,
                                                       aclDilationInfo);

    // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonDepthwiseConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         GetGuid());

    m_pDepthwiseConvolutionLayer->prepare();
}

void NeonDepthwiseConvolutionWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_NAME_GUID("NeonDepthwiseConvolutionWorkload_Execute");

    m_pDepthwiseConvolutionLayer->run();
}

} //namespace armnn
