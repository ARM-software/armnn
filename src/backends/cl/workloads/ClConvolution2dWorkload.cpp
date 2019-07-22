//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClConvolution2dWorkload.hpp"

#include "ClWorkloadUtils.hpp"

#include <cl/ClLayerSupport.hpp>
#include <cl/ClTensorHandle.hpp>
#include <cl/ClLayerSupport.hpp>
#include <aclCommon/ArmComputeUtils.hpp>
#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include <arm_compute/runtime/CL/functions/CLConvolutionLayer.h>

namespace armnn
{
using namespace armcomputetensorutils;

arm_compute::Status ClConvolution2dWorkloadValidate(const TensorInfo& input,
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

    return arm_compute::CLConvolutionLayer::validate(&aclInputInfo,
                                                     &aclWeightsInfo,
                                                     optionalAclBiasesInfo,
                                                     &aclOutputInfo,
                                                     layerInfo,
                                                     arm_compute::WeightsInfo(),
                                                     aclDilationInfo);
}

ClConvolution2dWorkload::ClConvolution2dWorkload(const Convolution2dQueueDescriptor& descriptor,
    const WorkloadInfo& info, std::shared_ptr<arm_compute::MemoryManagerOnDemand>& memoryManager)
    : BaseWorkload<Convolution2dQueueDescriptor>(descriptor, info)
    , m_ConvolutionLayer(memoryManager)
{
    // todo: check tensor shapes match.
    const TensorInfo& weightInfo = m_Data.m_Weight->GetTensorInfo();

    m_KernelTensor = std::make_unique<arm_compute::CLTensor>();
    BuildArmComputeTensor(*m_KernelTensor, weightInfo, m_Data.m_Parameters.m_DataLayout);

    const arm_compute::Size2D aclDilationInfo = BuildArmComputeSize2D(m_Data.m_Parameters.m_DilationX,
                                                                      m_Data.m_Parameters.m_DilationY);

    if (m_Data.m_Parameters.m_BiasEnabled)
    {
        m_BiasTensor = std::make_unique<arm_compute::CLTensor>();
        BuildArmComputeTensor(*m_BiasTensor, m_Data.m_Bias->GetTensorInfo(), m_Data.m_Parameters.m_DataLayout);
    }

    m_Data.ValidateInputsOutputs("ClConvolution2dWorkload", 1, 1);

    arm_compute::ICLTensor& input  = static_cast<IClTensorHandle*>(m_Data.m_Inputs[0])->GetTensor();
    arm_compute::ICLTensor& output = static_cast<IClTensorHandle*>(m_Data.m_Outputs[0])->GetTensor();

    arm_compute::DataLayout aclDataLayout = ConvertDataLayout(m_Data.m_Parameters.m_DataLayout);
    input.info()->set_data_layout(aclDataLayout);
    output.info()->set_data_layout(aclDataLayout);

    arm_compute::PadStrideInfo padStrideInfo = BuildArmComputePadStrideInfo(m_Data.m_Parameters);

    m_ConvolutionLayer.configure(&input,
                                 m_KernelTensor.get(),
                                 m_BiasTensor.get(),
                                 &output,
                                 padStrideInfo,
                                 arm_compute::WeightsInfo(),
                                 aclDilationInfo);

    InitializeArmComputeClTensorData(*m_KernelTensor, m_Data.m_Weight);

    if (m_BiasTensor)
    {
        InitializeArmComputeClTensorData(*m_BiasTensor, m_Data.m_Bias);
    }

    // Force Compute Library to perform the necessary copying and reshaping, after which
    // delete all the input tensors that will no longer be needed
    m_ConvolutionLayer.prepare();
    FreeUnusedTensors();
}

void ClConvolution2dWorkload::Execute() const
{
    ARMNN_SCOPED_PROFILING_EVENT_CL("ClConvolution2dWorkload_Execute");
    RunClFunction(m_ConvolutionLayer, CHECK_LOCATION());
}

void ClConvolution2dWorkload::FreeUnusedTensors()
{
    FreeTensorIfUnused(m_KernelTensor);
    FreeTensorIfUnused(m_BiasTensor);
}

} //namespace armnn
