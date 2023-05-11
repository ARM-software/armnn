//
// Copyright © 2017,2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvolution2dWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>

#include <armnn/Types.hpp>
#include <Half.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonConvolution2dWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const Convolution2dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      bool isFastMathEnabled,
                                                      const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);
    aclWeightsInfo.set_are_values_constant(weights.IsConstant());

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                      descriptor.m_DilationY);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());
        // Same for bias as weights. We don't currently support non const.
        if (!biases.value().IsConstant())
        {
            return arm_compute::Status{arm_compute::ErrorCode::RUNTIME_ERROR,
                                       "ArmNN NeonConvolution2dWorkload does not support non constant bias."};
        }
        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        aclBiasesInfo.set_are_values_constant(biases.value().IsConstant());
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo layerInfo = BuildArmComputePadStrideInfo(descriptor);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertActivationDescriptorToAclActivationLayerInfo(
            activationDescriptor);

    return arm_compute::NEConvolutionLayer::validate(&aclInputInfo,
                                                     &aclWeightsInfo,
                                                     optionalAclBiasesInfo,
                                                     &aclOutputInfo,
                                                     layerInfo,
                                                     arm_compute::WeightsInfo(),
                                                     aclDilationInfo,
                                                     activationInfo,
                                                     isFastMathEnabled);
}

NeonConvolution2dWorkload::NeonConvolution2dWorkload(
    const Convolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info,
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
    const bool isFastMathEnabled)
    : NeonBaseWorkload<Convolution2dQueueDescriptor>(descriptor, info)
{
    using arm_compute::NEConvolutionLayer;

    uint32_t numInputs = m_Data.m_Parameters.m_BiasEnabled ? 3: 2;
    m_Data.ValidateInputsOutputs("NeonConvolution2dWorkload", numInputs, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_KernelTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_KernelTensor, info.m_InputTensorInfos[1], m_Data.m_Parameters.m_DataLayout);
    m_KernelTensor->info()->set_are_values_constant(info.m_InputTensorInfos[1].IsConstant());
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_BiasTensor, info.m_InputTensorInfos[2], m_Data.m_Parameters.m_DataLayout);
        m_BiasTensor->info()->set_are_values_constant(info.m_InputTensorInfos[2].IsConstant());
        // We do not support dynamic bias
        ARMNN_ASSERT(info.m_InputTensorInfos[2].IsConstant() == true);
    }

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(m_Data.m_Parameters.m_DilationX,
                                                                      m_Data.m_Parameters.m_DilationY);

    const arm_compute::ActivationLayerInfo activationInfo = ConvertAdditionalInfoToAclActivationLayerInfo(descriptor);

    auto convolutionLayer = std::make_unique<arm_compute::NEConvolutionLayer>(memoryManager);
    convolutionLayer->configure(&input,
                                m_KernelTensor.get(),
                                m_BiasTensor.get(),
                                &output,
                                padStrideInfo,
                                arm_compute::WeightsInfo(),
                                aclDilationInfo,
                                activationInfo,
                                isFastMathEnabled);

    m_ConvolutionMethod =
        convolutionLayer->get_convolution_method(input.info(),
                                                 m_KernelTensor->info(),
                                                 output.info(),
                                                 padStrideInfo,
                                                 arm_compute::WeightsInfo(),
                                                 aclDilationInfo,
                                                 activationInfo,
                                                 isFastMathEnabled);

    // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;
    detailsInfo.m_WeightsTensorInfo = armnn::Optional<armnn::TensorInfo>(info.m_InputTensorInfos[1]);
    detailsInfo.m_ConvolutionMethod = armnn::Optional<std::string>(GetConvolutionMethodString(m_ConvolutionMethod));

    if (descriptor.m_Parameters.m_BiasEnabled)
    {
        detailsInfo.m_BiasTensorInfo = armnn::Optional<armnn::TensorInfo>(info.m_InputTensorInfos[2]);
    }

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         GetGuid());

    m_ConvolutionLayer.reset(convolutionLayer.release());

    ARMNN_ASSERT(m_ConvolutionLayer);
    m_KernelTensorInfo = info.m_InputTensorInfos[1];

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensorInfo = info.m_InputTensorInfos[2];
    }
}

void NeonConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonConvolution2dWorkload_Execute", this->GetGuid());
    // The constant tensors may not be fully in place until the workload is Executed
    if (!prepared)
    {
        InitializeArmComputeTensorData(*m_KernelTensor, m_KernelTensorInfo, m_Data.m_Inputs[1]);

        if (m_Data.m_Parameters.m_BiasEnabled)
        {
            InitializeArmComputeTensorData(*m_BiasTensor, m_BiasTensorInfo, m_Data.m_Inputs[2]);
        }
        m_ConvolutionLayer->prepare();
        FreeTensorIfUnused(m_KernelTensor);
        FreeTensorIfUnused(m_BiasTensor);
        prepared = true;
    }
    m_ConvolutionLayer->run();
}

arm_compute::ConvolutionMethod NeonConvolution2dWorkload::GetConvolutionMethod() const
{
    return m_ConvolutionMethod;
}

} //namespace armnn
