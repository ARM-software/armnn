//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvolution3dWorkload.hpp"

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <neon/workloads/NeonWorkloadUtils.hpp>

#include <arm_compute/runtime/NEON/functions/NEConv3D.h>

#include <armnn/Types.hpp>
#include <Half.hpp>

namespace armnn
{

using namespace armcomputetensorutils;

arm_compute::Status NeonConvolution3dWorkloadValidate(const TensorInfo& input,
                                                      const TensorInfo& output,
                                                      const Convolution3dDescriptor& descriptor,
                                                      const TensorInfo& weights,
                                                      const Optional<TensorInfo>& biases,
                                                      bool isFastMathEnabled,
                                                      const ActivationDescriptor* activationDescriptor)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);
    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;
    if (descriptor.m_BiasEnabled)
    {
        ARMNN_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);

    const arm_compute::Conv3dInfo aclConv3DInfo = ComputeConv3DInfo(descriptor,
                                                                    isFastMathEnabled,
                                                                    activationDescriptor);

    return arm_compute::NEConv3D::validate(&aclInputInfo,
                                           &aclWeightsInfo,
                                           optionalAclBiasesInfo,
                                           &aclOutputInfo,
                                           aclConv3DInfo);
}

NeonConvolution3dWorkload::NeonConvolution3dWorkload(const Convolution3dQueueDescriptor& descriptor,
                                                     const WorkloadInfo& info,
                                                     std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager,
                                                     const bool isFastMathEnabled)
    : NeonBaseWorkload<Convolution3dQueueDescriptor>(descriptor, info)
{
    IgnoreUnused(memoryManager);

    using arm_compute::NEConv3D;
    uint32_t numInputs = m_Data.m_Parameters.m_BiasEnabled ? 3: 2;
    m_Data.ValidateInputsOutputs("NeonConvolution3dWorkload", numInputs, 1);

    arm_compute::ITensor& input = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& weights = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Inputs[1])->GetTensor();
    arm_compute::ITensor* biasesPtr = nullptr;
    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        biasesPtr = &PolymorphicDowncast<IAclTensorHandle *>(m_Data.m_Inputs[2])->GetTensor();
    }
    arm_compute::ITensor& output = PolymorphicDowncast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    weights.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    const arm_compute::Conv3dInfo aclConv3DInfo = ComputeConv3DInfo(descriptor, isFastMathEnabled);

    auto convolutionLayer = std::make_unique<arm_compute::NEConv3D>();
    convolutionLayer->configure(&input,
                                &weights,
                                biasesPtr,
                                &output,
                                aclConv3DInfo);

    // Add details for profiling output
    WorkloadInfo detailsInfo;

    detailsInfo.m_InputTensorInfos = info.m_InputTensorInfos;
    detailsInfo.m_OutputTensorInfos = info.m_OutputTensorInfos;

    // Report Profiling Details
    ARMNN_REPORT_PROFILING_WORKLOAD_DESC("NeonConvolution3dWorkload_Construct",
                                         descriptor.m_Parameters,
                                         detailsInfo,
                                         this->GetGuid());

    m_ConvolutionLayer.reset(convolutionLayer.release());

    ARMNN_ASSERT(m_ConvolutionLayer);

    m_ConvolutionLayer->prepare();
}

void NeonConvolution3dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON_GUID("NeonConvolution3dWorkload_Execute", this->GetGuid());
    m_ConvolutionLayer->run();
}

} //namespace armnn
