//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NeonTransposeConvolution2dWorkload.hpp"

#include "NeonWorkloadUtils.hpp"

#include <Profiling.hpp>

#include <armnn/Types.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

#include <aclCommon/ArmComputeTensorUtils.hpp>

#include <armnn/backends/TensorHandle.hpp>

#include <neon/workloads/NeonWorkloadUtils.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonTransposeConvolution2dWorkloadValidate(const TensorInfo& input,
                                                               const TensorInfo& output,
                                                               const TransposeConvolution2dDescriptor& descriptor,
                                                               const TensorInfo& weights,
                                                               const Optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo   = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo  = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo layerInfo = BuildArmComputePadStrideInfo(descriptor);

    return arm_compute::NEDeconvolutionLayer::validate(&aclInputInfo,
                                                       &aclWeightsInfo,
                                                       optionalAclBiasesInfo,
                                                       &aclOutputInfo,
                                                       layerInfo);
}

NeonTransposeConvolution2dWorkload::NeonTransposeConvolution2dWorkload(
    const TransposeConvolution2dQueueDescriptor& descriptor, const WorkloadInfo& info,
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : NeonBaseWorkload<TransposeConvolution2dQueueDescriptor>(descriptor, info)
{
    m_Data.ValidateInputsOutputs("NeonTransposeConvolution2dWorkload", 1, 1);

    arm_compute::ITensor& input  = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    m_KernelTensor = std::make_unique<arm_compute::Tensor>();
    BuildArmComputeTensor(*m_KernelTensor, m_Data.m_Weight->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::Tensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);

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
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonTransposeConvolution2dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    m_Layer = std::make_unique<arm_compute::NEDeconvolutionLayer>(memoryManager);
    m_Layer->configure(&input, m_KernelTensor.get(), m_BiasTensor.get(), &output, padStrideInfo);

    ARMNN_ASSERT(m_Layer);

    InitializeArmComputeTensorData(*m_KernelTensor, m_Data.m_Weight);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        InitializeArmComputeTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    m_Layer->prepare();
    FreeUnusedTensors();
}

void NeonTransposeConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonTransposeConvolution2dWorkload_Execute", this->GetGuid());
    m_Layer->run();
}

void NeonTransposeConvolution2dWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

} // namespace armnn
