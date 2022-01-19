//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClDepthwiseConvolutionWorkload.hpp"

#include <ResolveType.hpp>
#include "ClWorkloadUtils.hpp"

#include <armnn/Exceptions.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <cl/ClTensorHandle.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <backendsCommon/WorkloadUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>

#include <arm_compute/runtime/CL/functions/CLDepthwiseConvolutionLayer.h>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status ClDepthwiseConvolutionWorkloadValidate(const TensorInfo& input,
                                                           const TensorInfo& output,
                                                           const DepthwiseConvolution2dDescriptor& descriptor,
                                                           const TensorInfo& weights,
                                                           const Optional<TensorInfo>& biases,
                                                           const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInputInfo  = BuildArmComputeTensorInfo(input,  descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    // ArmNN's weight format is usually [ M, I, H, W ] but for depthwise its [ 1, H, W, I*M]
    // Permute to [ 1, I * M, H, W ] (if NCHW) as required by the compute library
    unsigned int aclDepthMultiplier;
    TensorInfo weightsPermuted;
    std::tie(weightsPermuted, aclDepthMultiplier) = Convert1HWOTensorInfoToAcl(weights, input,descriptor.m_DataLayout);

    // Convert the weights into the compute library format
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weightsPermuted, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    const arm_compute::PadStrideInfo aclPadStrideInfo = BuildArmComputePadStrideInfo(descriptor);
    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(
            descriptor.m_DilationX,
            descriptor.m_DilationY);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::CLDepthwiseConvolutionLayer::validate(&aclInputInfo,
                                                              &aclWeightsInfo,
                                                              optionalAclBiasesInfo,
                                                              &aclOutputInfo,
                                                              aclPadStrideInfo,
                                                              aclDepthMultiplier,
                                                              activationInfo,
                                                              aclDilationInfo);

}

ClDepthwiseConvolutionWorkload::ClDepthwiseConvolutionWorkload(
    const DepthwiseConvolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    const arm_compute::CLCompileContext& clCompileContext)
    : ClBaseWorkload<DepthwiseConvolution2dQueueDescriptor>(descriptor, info)
{
    // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Weight->GetTensorInfo());
    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(descriptor.m_Bias->GetTensorInfo());
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("ClDepthwiseConvolutionWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    // ArmNN's weight format is usually [ M, I, H, W ] but for depthwise its [ 1, H, W, I*M]
    // Permute to [ 1, I * M, H, W ] (if NCHW), as required by the compute library
    ConstTensor weightPermuted;
    unsigned int depthMultiplier;
    std::unique_ptr<unsigned char[]> permuteBuffer(new unsigned char[m_Data.m_Weight->GetTensorInfo().GetNumBytes()]);
    std::tie(weightPermuted, depthMultiplier) = Convert1HWOTensorToAcl(m_Data.m_Weight,
                                                                       info.m_InputTensorInfos[0],
                                                                       m_Data.m_Parameters.m_DataLayout,
                                                                       permuteBuffer.get());

    // Convert the weights into the compute library format
    m_KernelTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_KernelTensor, weightPermuted.GetInfo(), m_Data.m_Parameters.m_DataLayout);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(
                m_Data.m_Parameters.m_DilationX,
                m_Data.m_Parameters.m_DilationY);


    std::string name = std::string("ClDepthwiseConvolutionWorkload");
    m_Data.ValidateInputsOutputs(name, 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    m_DepthwiseConvolutionLayer = std::make_unique<arm_compute::CLDepthwiseConvolutionLayer>();

    {
        ARMNN_SCOPED_PROFILING_EVENT(Compute::Undefined, "ClDepthwiseConvolutionWorkload_configure");
        static_cast<arm_compute::CLDepthwiseConvolutionLayer*>(m_DepthwiseConvolutionLayer.get())->configure(
                clCompileContext,
                &input,
                m_KernelTensor.get(),
                m_BiasTensor.get(),
                &output,
                padStrideInfo,
                depthMultiplier,
                activationInfo,
                aclDilationInfo);
    }
    ARMNN_ASSERT(m_DepthwiseConvolutionLayer);

    ScopedTensorHandle weightsPermutedHandle(weightPermuted);
    InitializeArmComputeClTensorData(*m_KernelTensor, &weightsPermutedHandle);

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
    ARMNN_SCOPED_PROFILING_EVENT_CL_GUID("ClDepthwiseConvolutionWorkload_Execute", this->GetGuid());
    ARMNN_ASSERT(m_DepthwiseConvolutionLayer);

    RunClFunction(*m_DepthwiseConvolutionLayer, CHECK_LOCATION());
}

} // namespace armnn
