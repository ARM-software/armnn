//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonConvolution2dWorkload.hpp"

#include <backendsCommon/CpuTensorHandle.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
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
    const Optional<TensorInfo>& biases)
{
    const arm_compute::TensorInfo aclInputInfo = BuildArmComputeTensorInfo(input, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclOutputInfo = BuildArmComputeTensorInfo(output, descriptor.m_DataLayout);
    const arm_compute::TensorInfo aclWeightsInfo = BuildArmComputeTensorInfo(weights, descriptor.m_DataLayout);

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(descriptor.m_DilationX,
                                                                      descriptor.m_DilationY);

    arm_compute::TensorInfo aclBiasesInfo;
    arm_compute::TensorInfo *optionalAclBiasesInfo = nullptr;

    if (descriptor.m_BiasEnabled)
    {
        BOOST_ASSERT(biases.has_value());

        aclBiasesInfo = BuildArmComputeTensorInfo(biases.value(), descriptor.m_DataLayout);
        optionalAclBiasesInfo = &aclBiasesInfo;
    }

    arm_compute::PadStrideInfo layerInfo = BuildArmComputePadStrideInfo(descriptor);

    return arm_compute::NEConvolutionLayer::validate(&aclInputInfo,
                                                     &aclWeightsInfo,
                                                     optionalAclBiasesInfo,
                                                     &aclOutputInfo,
                                                     layerInfo,
                                                     arm_compute::WeightsInfo(),
                                                     aclDilationInfo);
}

NeonConvolution2dWorkload::NeonConvolution2dWorkload(
    const Convolution2dQueueDescriptor& descriptor, const WorkloadInfo& info,
    std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : BaseWorkload<Convolution2dQueueDescriptor>(descriptor, info)
{
    using arm_compute::NEDirectConvolutionLayer;

    m_Data.ValidateInputsOutputs("NeonConvolution2dWorkload", 1, 1);

    // todo: check tensor shapes match.

    arm_compute::ITensor& input = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ITensor& output = boost::polymorphic_downcast<IAclTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

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

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(m_Data.m_Parameters.m_DilationX,
                                                                      m_Data.m_Parameters.m_DilationY);

    auto convolutionLayer = std::make_unique<arm_compute::NEConvolutionLayer>(memoryManager);
    convolutionLayer->configure(&input,
                                m_KernelTensor.get(),
                                m_BiasTensor.get(),
                                &output,
                                padStrideInfo,
                                arm_compute::WeightsInfo(),
                                aclDilationInfo);

    m_ConvolutionLayer.reset(convolutionLayer.release());

    BOOST_ASSERT(m_ConvolutionLayer);

    InitializeArmComputeTensorData(*m_KernelTensor, m_Data.m_Weight);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        InitializeArmComputeTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    m_ConvolutionLayer->prepare();
    FreeUnusedTensors();
}

void NeonConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_NEON("NeonConvolution2dWorkload_Execute");
    m_ConvolutionLayer->run();
}

void NeonConvolution2dWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

} //namespace armnn
